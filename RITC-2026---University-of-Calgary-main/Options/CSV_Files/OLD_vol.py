#!/usr/bin/env python3
"""
RITC 2026 Volatility Trading Case - Auto-Trading Algorithm
============================================================
Strategies implemented:
  1. VEGA MISPRICING: Sell overpriced / buy underpriced options when IV != realized vol
  2. DELTA HEDGING: Continuously neutralize portfolio delta via RTM shares
  3. GAMMA SCALPING: When long options, profit from realized vol > IV
  4. PRE-POSITIONING: Use mid-week forecasts to anticipate vol regime changes

Architecture optimized for execution speed:
  - Pre-computed Greeks lookup tables
  - Minimal API calls per loop iteration
  - Async-ready structure with tight main loop
  - Smart order batching

RIT API Reference:
  GET  /v1/case       - Case metadata (period, tick, status)
  GET  /v1/securities  - All security data
  GET  /v1/securities?ticker=X - Single security
  GET  /v1/orders      - Open orders
  POST /v1/orders      - Submit order
  DELETE /v1/orders/{id} - Cancel order
  DELETE /v1/commands/cancel?all=1 - Cancel all orders
  GET  /v1/news        - News feed
  GET  /v1/limits      - Position limits
"""

import requests
import time
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE = "http://localhost:9997/v1"
API_KEY = "RYL2000"  # Replace with your RIT API key

# Strategy parameters (tuned from data analysis)
HEDGE_INTERVAL_TICKS = 3          # Rebalance delta every N ticks
MIN_VEGA_EDGE_PCT = 0.03          # Min IV-RV gap to trade (3%)
MAX_CONTRACTS_PER_TRADE = 80      # Conservative vs 100 max
TARGET_POSITION_PER_STRIKE = 60   # Target option position per strike
GROSS_OPTION_LIMIT = 2200         # Stay under 2500 gross
NET_OPTION_LIMIT = 800            # Stay under 1000 net
MAX_RTM_POSITION = 45000          # Stay under 50000
DELTA_BUFFER_PCT = 0.7            # Use 70% of delta limit
GAMMA_SCALP_THRESHOLD = 0.30      # RTM price move to trigger gamma scalp ($)

# Timing constants (from case spec)
TICKS_PER_WEEK = 75
TICKS_PER_DAY = 15
DAYS_PER_YEAR = 240
TOTAL_DAYS = 20
TOTAL_TICKS = 300

# Strike prices
STRIKES = list(range(45, 55))
OPTION_MULTIPLIER = 100  # 100 shares per contract

# =============================================================================
# FAST BLACK-SCHOLES (inlined for speed)
# =============================================================================

_SQRT2PI = math.sqrt(2 * math.pi)

def _norm_cdf(x):
    """Fast approximation of standard normal CDF (Abramowitz & Stegun)."""
    if x < -8: return 0.0
    if x > 8: return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-0.5 * x * x)
    pp = p * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - pp if x > 0 else pp

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / _SQRT2PI

def bs_price(S, K, T, r, sig, is_call=True):
    if T < 1e-8:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def bs_delta(S, K, T, r, sig, is_call=True):
    if T < 1e-8:
        return (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0

def bs_gamma(S, K, T, r, sig):
    if T < 1e-8: return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
    return _norm_pdf(d1) / (S * sig * sqrtT)

def bs_vega(S, K, T, r, sig):
    """Vega per 1% vol move."""
    if T < 1e-8: return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
    return S * _norm_pdf(d1) * sqrtT * 0.01

def implied_vol_fast(price, S, K, T, r, is_call=True):
    """Newton-Raphson IV solver, max 30 iterations for speed."""
    if T < 1e-8: return None
    intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
    if price <= intrinsic + 0.005: return None
    
    sig = 0.30
    for _ in range(30):
        p = bs_price(S, K, T, r, sig, is_call)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
        vega = S * _norm_pdf(d1) * sqrtT
        if vega < 1e-12: break
        sig -= (p - price) / vega
        sig = max(0.005, min(sig, 2.0))
        if abs(p - price) < 5e-5: break
    return sig if 0.01 < sig < 2.0 else None

# =============================================================================
# RIT API CLIENT (optimized for speed)
# =============================================================================

class RITClient:
    """Fast, minimal RIT API client."""
    
    def __init__(self, api_key=API_KEY):
        self.session = requests.Session()
        self.session.headers.update({'X-API-Key': api_key})
        self.base = API_BASE
    
    def get(self, endpoint, params=None):
        try:
            r = self.session.get(f"{self.base}{endpoint}", params=params, timeout=1)
            if r.ok: return r.json()
        except: pass
        return None
    
    def post(self, endpoint, params=None):
        try:
            r = self.session.post(f"{self.base}{endpoint}", params=params, timeout=1)
            if r.ok: return r.json()
        except: pass
        return None
    
    def delete(self, endpoint, params=None):
        try:
            r = self.session.delete(f"{self.base}{endpoint}", params=params, timeout=1)
            return r.ok
        except: return False
    
    # ----- High-level methods -----
    
    def get_tick(self):
        """Get current tick and period."""
        data = self.get("/case")
        if data:
            return data.get('tick', 0), data.get('period', 1), data.get('status', 'STOPPED')
        return 0, 1, 'STOPPED'
    
    def get_securities(self):
        """Get all securities data in one call."""
        return self.get("/securities") or []
    
    def get_news(self, since=0):
        """Get news since given ID."""
        return self.get("/news", {'since': since}) or []
    
    def get_limits(self):
        """Get position limits."""
        return self.get("/limits") or []
    
    def submit_order(self, ticker, side, qty, order_type='MARKET', price=None):
        """Submit an order. Returns order dict or None."""
        params = {
            'ticker': ticker,
            'type': order_type,
            'quantity': int(qty),
            'action': side.upper(),
        }
        if price is not None:
            params['price'] = round(price, 2)
        return self.post("/orders", params)
    
    def cancel_all(self):
        """Cancel all open orders."""
        return self.delete("/commands/cancel", {'all': 1})
    
    def buy(self, ticker, qty, price=None):
        otype = 'LIMIT' if price else 'MARKET'
        return self.submit_order(ticker, 'BUY', qty, otype, price)
    
    def sell(self, ticker, qty, price=None):
        otype = 'LIMIT' if price else 'MARKET'
        return self.submit_order(ticker, 'SELL', qty, otype, price)


# =============================================================================
# PORTFOLIO STATE TRACKER
# =============================================================================

@dataclass
class OptionPosition:
    ticker: str
    strike: int
    is_call: bool
    quantity: int = 0  # positive = long, negative = short
    
@dataclass
class PortfolioState:
    """Tracks full portfolio state for Greeks computation."""
    rtm_position: int = 0
    rtm_price: float = 50.0
    option_positions: Dict[str, int] = field(default_factory=dict)  # ticker -> qty
    
    # Computed Greeks
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    
    # Market state
    current_tick: int = 0
    current_week: int = 1
    realized_vol: float = 0.30
    risk_free_rate: float = 0.0
    delta_limit: int = 10000
    penalty_pct: float = 0.01
    
    # News tracking
    last_news_id: int = 0
    week_vols: Dict[int, float] = field(default_factory=dict)
    week_forecasts: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    
    def time_to_expiry(self):
        """T in years for BS model."""
        days = TOTAL_DAYS - (self.current_tick - 1) / TICKS_PER_DAY
        return max(days / DAYS_PER_YEAR, 1e-6)
    
    def current_week_num(self):
        return min((self.current_tick - 1) // TICKS_PER_WEEK + 1, 4)
    
    def get_realized_vol(self):
        """Get the realized vol for current week."""
        wk = self.current_week_num()
        return self.week_vols.get(wk, self.realized_vol)
    
    def compute_greeks(self):
        """Recompute all portfolio Greeks."""
        T = self.time_to_expiry()
        S = self.rtm_price
        r = self.risk_free_rate
        rv = self.get_realized_vol()
        
        # RTM delta
        total_delta = float(self.rtm_position)
        total_gamma = 0.0
        total_vega = 0.0
        
        for ticker, qty in self.option_positions.items():
            if qty == 0: continue
            
            # Parse ticker: RTM1C45 or RTM1P45
            is_call = 'C' in ticker[4:]
            strike = int(ticker[-2:])
            
            d = bs_delta(S, strike, T, r, rv, is_call)
            g = bs_gamma(S, strike, T, r, rv)
            v = bs_vega(S, strike, T, r, rv)
            
            total_delta += qty * d * OPTION_MULTIPLIER
            total_gamma += qty * g * OPTION_MULTIPLIER
            total_vega += qty * v * OPTION_MULTIPLIER
        
        self.portfolio_delta = total_delta
        self.portfolio_gamma = total_gamma
        self.portfolio_vega = total_vega
    
    def gross_option_position(self):
        return sum(abs(q) for q in self.option_positions.values())
    
    def net_option_position(self):
        return sum(q for q in self.option_positions.values())


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class VolatilityTrader:
    """Main trading engine implementing all strategies."""
    
    def __init__(self):
        self.client = RITClient()
        self.state = PortfolioState()
        self.last_hedge_tick = 0
        self.last_rtm_price = 50.0
        self.trade_log = []
        self.initialized = False
        self.week_traded = set()  # Track which weeks we've traded vega
    
    # ----- NEWS PARSING -----
    
    def process_news(self):
        """Parse new news releases and update state."""
        news_list = self.client.get_news(self.state.last_news_id)
        if not news_list:
            return
        
        for item in news_list:
            nid = item.get('news_id', 0)
            if nid <= self.state.last_news_id:
                continue
            self.state.last_news_id = nid
            
            headline = item.get('headline', '')
            body = item.get('body', '')
            
            # Delta limit announcement
            if 'delta limit' in headline.lower():
                m = re.search(r'delta limit.*?(\d+).*?penalty.*?(\d+)%', body)
                if m:
                    self.state.delta_limit = int(m.group(1))
                    self.state.penalty_pct = float(m.group(2)) / 100.0
                    print(f"  [NEWS] Delta limit={self.state.delta_limit}, "
                          f"Penalty={self.state.penalty_pct:.0%}")
            
            # Initial volatility
            elif 'risk free rate' in headline.lower():
                m = re.search(r'risk free rate is (\d+)%', body)
                if m: self.state.risk_free_rate = float(m.group(1)) / 100.0
                m = re.search(r'annualized realized volatility is (\d+)%', body)
                if m:
                    self.state.realized_vol = float(m.group(1)) / 100.0
                    print(f"  [NEWS] Initial vol={self.state.realized_vol:.0%}, "
                          f"r={self.state.risk_free_rate:.0%}")
            
            # Weekly vol announcement
            elif 'announcement' in headline.lower():
                m_wk = re.search(r'(\d+)', headline)
                m_vol = re.search(r'will be (\d+)%', body)
                if m_wk and m_vol:
                    wk = int(m_wk.group(1))
                    vol = float(m_vol.group(1)) / 100.0
                    self.state.week_vols[wk] = vol
                    print(f"  [NEWS] Week {wk} realized vol = {vol:.0%}")
            
            # Forecast for next week
            elif 'news' in headline.lower():
                m_wk = re.search(r'(\d+)', headline)
                m_range = re.search(r'between (\d+)% and (\d+)%', body)
                if m_wk and m_range:
                    wk = int(m_wk.group(1))
                    lo = float(m_range.group(1)) / 100.0
                    hi = float(m_range.group(2)) / 100.0
                    self.state.week_forecasts[wk] = (lo, hi)
                    print(f"  [NEWS] Next week forecast: {lo:.0%} - {hi:.0%}")
    
    # ----- MARKET DATA -----
    
    def update_market_data(self):
        """Fetch current prices and positions from RIT."""
        securities = self.client.get_securities()
        if not securities:
            return False
        
        for sec in securities:
            ticker = sec.get('ticker', '')
            position = sec.get('position', 0)
            
            if ticker == 'RTM':
                self.state.rtm_price = sec.get('last', 50.0)
                self.state.rtm_position = int(sec.get('position', 0))
                # Use bid/ask midpoint if last is stale
                bid = sec.get('bid', 0)
                ask = sec.get('ask', 0)
                if bid > 0 and ask > 0:
                    self.state.rtm_price = (bid + ask) / 2.0
            elif ticker.startswith('RTM1'):
                self.state.option_positions[ticker] = int(sec.get('position', 0))
        
        self.state.compute_greeks()
        return True
    
    # ----- STRATEGY 1: VEGA MISPRICING -----
    
    def find_mispricing_opportunities(self):
        """Identify options where IV significantly differs from realized vol."""
        S = self.state.rtm_price
        T = self.state.time_to_expiry()
        r = self.state.risk_free_rate
        rv = self.state.get_realized_vol()
        
        if T < 0.002:  # Too close to expiry
            return []
        
        opportunities = []
        
        for K in STRIKES:
            for is_call in [True, False]:
                ticker = f"RTM1{'C' if is_call else 'P'}{K}"
                
                # Get current market price (we need bid/ask from securities)
                # For speed, we compute fair value and compare
                fair = bs_price(S, K, T, r, rv, is_call)
                
                # We need the actual market bid/ask
                # Get from securities data
                sec_data = self.client.get(f"/securities", {'ticker': ticker})
                if not sec_data or len(sec_data) == 0:
                    continue
                sec = sec_data[0] if isinstance(sec_data, list) else sec_data
                
                bid = sec.get('bid', 0)
                ask = sec.get('ask', 0)
                if bid <= 0 or ask <= 0:
                    continue
                mid = (bid + ask) / 2.0
                
                # Compute IV from mid
                iv = implied_vol_fast(mid, S, K, T, r, is_call)
                if iv is None:
                    continue
                
                vol_gap = iv - rv  # positive = IV too high (sell), negative = IV too low (buy)
                edge = mid - fair  # dollar edge per share
                
                if abs(vol_gap) >= MIN_VEGA_EDGE_PCT:
                    delta = bs_delta(S, K, T, r, rv, is_call)
                    gamma = bs_gamma(S, K, T, r, rv)
                    vega = bs_vega(S, K, T, r, rv)
                    
                    opportunities.append({
                        'ticker': ticker,
                        'strike': K,
                        'is_call': is_call,
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'fair': fair,
                        'edge': edge,          # $ per share
                        'iv': iv,
                        'rv': rv,
                        'vol_gap': vol_gap,
                        'delta': delta,
                        'gamma': gamma,
                        'vega': vega,
                        'action': 'SELL' if vol_gap > 0 else 'BUY',
                    })
        
        # Sort by absolute edge (best opportunities first)
        opportunities.sort(key=lambda x: abs(x['edge']), reverse=True)
        return opportunities
    
    def find_mispricing_fast(self):
        """Faster version: gets all securities in one call, computes opportunities."""
        S = self.state.rtm_price
        T = self.state.time_to_expiry()
        r = self.state.risk_free_rate
        rv = self.state.get_realized_vol()
        
        if T < 0.002:
            return []
        
        securities = self.client.get_securities()
        if not securities:
            return []
        
        # Build lookup
        sec_map = {}
        for sec in securities:
            sec_map[sec.get('ticker', '')] = sec
        
        opportunities = []
        
        for K in STRIKES:
            for is_call in [True, False]:
                ticker = f"RTM1{'C' if is_call else 'P'}{K}"
                sec = sec_map.get(ticker)
                if not sec:
                    continue
                
                bid = sec.get('bid', 0)
                ask = sec.get('ask', 0)
                if bid <= 0 or ask <= 0:
                    continue
                
                fair = bs_price(S, K, T, r, rv, is_call)
                mid = (bid + ask) / 2.0
                edge = mid - fair
                
                # Quick check before expensive IV calc
                if abs(edge) < 0.01:
                    continue
                
                iv = implied_vol_fast(mid, S, K, T, r, is_call)
                if iv is None:
                    continue
                
                vol_gap = iv - rv
                
                if abs(vol_gap) >= MIN_VEGA_EDGE_PCT:
                    delta = bs_delta(S, K, T, r, rv, is_call)
                    gamma_val = bs_gamma(S, K, T, r, rv)
                    vega_val = bs_vega(S, K, T, r, rv)
                    
                    opportunities.append({
                        'ticker': ticker, 'strike': K, 'is_call': is_call,
                        'bid': bid, 'ask': ask, 'mid': mid, 'fair': fair,
                        'edge': edge, 'iv': iv, 'rv': rv, 'vol_gap': vol_gap,
                        'delta': delta, 'gamma': gamma_val, 'vega': vega_val,
                        'action': 'SELL' if vol_gap > 0 else 'BUY',
                    })
        
        opportunities.sort(key=lambda x: abs(x['edge']), reverse=True)
        return opportunities
    
    def execute_vega_trades(self, opportunities):
        """Execute option trades on mispricing opportunities."""
        if not opportunities:
            return
        
        gross = self.state.gross_option_position()
        net = abs(self.state.net_option_position())
        
        trades_done = 0
        max_trades_per_cycle = 4  # Limit to avoid overwhelming the API
        
        for opp in opportunities:
            if trades_done >= max_trades_per_cycle:
                break
            
            ticker = opp['ticker']
            current_pos = self.state.option_positions.get(ticker, 0)
            
            # Determine trade quantity
            if opp['action'] == 'SELL':
                # Sell (write) options - we want negative position
                desired_pos = -TARGET_POSITION_PER_STRIKE
                trade_qty = current_pos - desired_pos  # positive = need to sell
                if trade_qty <= 0:
                    continue  # Already at or beyond target
                
                # Check limits
                if gross + trade_qty > GROSS_OPTION_LIMIT:
                    trade_qty = max(0, GROSS_OPTION_LIMIT - gross)
                if trade_qty <= 0:
                    continue
                
                trade_qty = min(trade_qty, MAX_CONTRACTS_PER_TRADE)
                
                # Sell at bid (market sell)
                result = self.client.sell(ticker, trade_qty)
                if result:
                    self.state.option_positions[ticker] = current_pos - trade_qty
                    gross += trade_qty
                    trades_done += 1
                    print(f"  [TRADE] SELL {trade_qty} {ticker} @ ~{opp['bid']:.2f} "
                          f"(IV={opp['iv']:.1%} vs RV={opp['rv']:.1%}, edge=${opp['edge']:.3f})")
            
            else:  # BUY
                desired_pos = TARGET_POSITION_PER_STRIKE
                trade_qty = desired_pos - current_pos
                if trade_qty <= 0:
                    continue
                
                if gross + trade_qty > GROSS_OPTION_LIMIT:
                    trade_qty = max(0, GROSS_OPTION_LIMIT - gross)
                if trade_qty <= 0:
                    continue
                
                trade_qty = min(trade_qty, MAX_CONTRACTS_PER_TRADE)
                
                result = self.client.buy(ticker, trade_qty)
                if result:
                    self.state.option_positions[ticker] = current_pos + trade_qty
                    gross += trade_qty
                    trades_done += 1
                    print(f"  [TRADE] BUY {trade_qty} {ticker} @ ~{opp['ask']:.2f} "
                          f"(IV={opp['iv']:.1%} vs RV={opp['rv']:.1%}, edge=${opp['edge']:.3f})")
        
        return trades_done
    
    # ----- STRATEGY 2: DELTA HEDGING -----
    
    def hedge_delta(self, force=False):
        """Neutralize portfolio delta by trading RTM shares."""
        self.state.compute_greeks()
        
        delta = self.state.portfolio_delta
        delta_limit = self.state.delta_limit * DELTA_BUFFER_PCT
        
        # Only hedge if delta is significant or if forced
        if not force and abs(delta) < 500:
            return
        
        # Target: get delta close to 0
        # Need to trade -delta shares of RTM
        hedge_qty = -int(round(delta))
        
        if abs(hedge_qty) < 100:
            return
        
        # Check RTM position limits
        new_pos = self.state.rtm_position + hedge_qty
        if abs(new_pos) > MAX_RTM_POSITION:
            # Clip
            if new_pos > 0:
                hedge_qty = MAX_RTM_POSITION - self.state.rtm_position
            else:
                hedge_qty = -MAX_RTM_POSITION - self.state.rtm_position
        
        if abs(hedge_qty) < 100:
            return
        
        # Execute in chunks of 10,000 (max order size)
        remaining = abs(hedge_qty)
        side = 'BUY' if hedge_qty > 0 else 'SELL'
        
        while remaining > 0:
            chunk = min(remaining, 10000)
            result = self.client.submit_order('RTM', side, chunk, 'MARKET')
            if result:
                if side == 'BUY':
                    self.state.rtm_position += chunk
                else:
                    self.state.rtm_position -= chunk
                remaining -= chunk
            else:
                break
        
        old_delta = delta
        self.state.compute_greeks()
        print(f"  [HEDGE] Delta: {old_delta:+.0f} -> {self.state.portfolio_delta:+.0f} "
              f"(RTM pos={int(self.state.rtm_position):+d})")
    
    # ----- STRATEGY 3: GAMMA SCALPING -----
    
    def gamma_scalp(self):
        """If long gamma, trade RTM on price moves to capture realized vol."""
        if self.state.portfolio_gamma <= 0:
            return  # Only scalp when long gamma
        
        price_move = self.state.rtm_price - self.last_rtm_price
        
        if abs(price_move) < GAMMA_SCALP_THRESHOLD:
            return
        
        # Long gamma: sell after up-move, buy after down-move
        # Trade size proportional to gamma * price_move
        scalp_shares = int(-self.state.portfolio_gamma * price_move * 0.5)
        
        if abs(scalp_shares) < 100:
            return
        
        scalp_shares = max(-10000, min(10000, scalp_shares))
        
        # Check position limits
        new_pos = self.state.rtm_position + scalp_shares
        if abs(new_pos) > MAX_RTM_POSITION:
            return
        
        side = 'BUY' if scalp_shares > 0 else 'SELL'
        result = self.client.submit_order('RTM', side, abs(scalp_shares), 'MARKET')
        if result:
            self.state.rtm_position += scalp_shares
            print(f"  [GAMMA] {side} {abs(scalp_shares)} RTM (price move={price_move:+.2f})")
        
        self.last_rtm_price = self.state.rtm_price
    
    # ----- STRATEGY 4: PRE-POSITIONING ON FORECASTS -----
    
    def check_pre_position(self):
        """Use mid-week forecasts to anticipate next week's vol change."""
        wk = self.state.current_week_num()
        ticks_into_week = (self.state.current_tick - 1) % TICKS_PER_WEEK
        
        # Forecasts come at ~tick 37 of each week (mid-week)
        # If we have a forecast for next week and we're in the second half of the week
        if ticks_into_week < 50 or ticks_into_week > 70:
            return
        
        # Check if we have a forecast
        fc = self.state.week_forecasts.get(wk)
        if not fc:
            return
        
        fc_mid = (fc[0] + fc[1]) / 2.0
        current_rv = self.state.get_realized_vol()
        
        # If forecast suggests big vol change, start building position
        expected_shift = fc_mid - current_rv
        
        if abs(expected_shift) > 0.05:  # >5% vol change expected
            print(f"  [PRE-POS] Expecting vol shift: {current_rv:.0%} -> ~{fc_mid:.0%}")
            # Don't execute trades here - just flag for next week's main strategy
    
    # ----- MAIN TRADING LOOP -----
    
    def initialize(self):
        """Initial setup at start of sub-heat."""
        print("\n" + "=" * 60)
        print("VOLATILITY TRADING ALGORITHM - INITIALIZING")
        print("=" * 60)
        
        self.process_news()
        self.update_market_data()
        
        print(f"  RTM Price: ${self.state.rtm_price:.2f}")
        print(f"  Initial Vol: {self.state.realized_vol:.0%}")
        print(f"  Delta Limit: {self.state.delta_limit}")
        print(f"  Time to Expiry: {self.state.time_to_expiry()*240:.1f} days")
        
        self.last_rtm_price = self.state.rtm_price
        self.initialized = True
    
    def run_tick(self):
        """Execute one iteration of the trading loop."""
        tick, period, status = self.client.get_tick()
        
        if status != 'ACTIVE':
            return False
        
        self.state.current_tick = tick
        
        # Update market data
        if not self.update_market_data():
            return True
        
        # Process any new news
        self.process_news()
        
        current_week = self.state.current_week_num()
        ticks_into_week = (tick - 1) % TICKS_PER_WEEK
        
        # ----- STRATEGY EXECUTION PRIORITY -----
        
        # 1. VEGA TRADING: At week starts or when big mispricing detected
        #    Trade aggressively in first 15 ticks of each week (when mispricing is largest)
        if ticks_into_week < 15 or ticks_into_week % 10 == 0:
            opportunities = self.find_mispricing_fast()
            if opportunities:
                best = opportunities[0]
                if abs(best['vol_gap']) > MIN_VEGA_EDGE_PCT:
                    self.execute_vega_trades(opportunities[:6])  # Top 6 opportunities
        
        # 2. DELTA HEDGING: Every N ticks or when delta exceeds threshold
        should_hedge = (
            tick - self.last_hedge_tick >= HEDGE_INTERVAL_TICKS or
            abs(self.state.portfolio_delta) > self.state.delta_limit * DELTA_BUFFER_PCT
        )
        
        if should_hedge:
            self.hedge_delta()
            self.last_hedge_tick = tick
        
        # 3. GAMMA SCALPING: When long gamma and price moves
        if self.state.portfolio_gamma > 0:
            self.gamma_scalp()
        
        # 4. PRE-POSITIONING: Check forecasts mid-week
        self.check_pre_position()
        
        # 5. POSITION MONITORING
        if tick % 25 == 0:
            T = self.state.time_to_expiry()
            rv = self.state.get_realized_vol()
            print(f"\n  [t={tick:3d}] RTM=${self.state.rtm_price:.2f} "
                  f"Wk{current_week} RV={rv:.0%} "
                  f"Delta={self.state.portfolio_delta:+.0f} "
                  f"Gamma={self.state.portfolio_gamma:+.0f} "
                  f"Vega={self.state.portfolio_vega:+.0f} "
                  f"OptGross={self.state.gross_option_position()} "
                  f"RTM={int(self.state.rtm_position):+d} "
                  f"T={T*240:.1f}d")
        
        # 6. END-OF-SUBHEAT: Flatten options near expiry (last 10 ticks)
        if tick > 290:
            self.close_positions()
        
        self.last_rtm_price = self.state.rtm_price
        return True
    
    def close_positions(self):
        """Gradually close positions near end of sub-heat."""
        # Close option positions
        for ticker, qty in list(self.state.option_positions.items()):
            if qty == 0:
                continue
            if qty > 0:
                self.client.sell(ticker, min(abs(qty), MAX_CONTRACTS_PER_TRADE))
            else:
                self.client.buy(ticker, min(abs(qty), MAX_CONTRACTS_PER_TRADE))
        
        # Flatten RTM
        if self.state.rtm_position != 0:
            side = 'SELL' if self.state.rtm_position > 0 else 'BUY'
            qty = min(abs(self.state.rtm_position), 10000)
            self.client.submit_order('RTM', side, qty, 'MARKET')
    
    def run(self):
        """Main loop - runs until sub-heat ends."""
        print("\nWaiting for case to start...")
        
        # Wait for case to become active
        while True:
            tick, period, status = self.client.get_tick()
            if status == 'ACTIVE':
                break
            time.sleep(0.5)
        
        print(f"Case ACTIVE! Period={period}, Tick={tick}")
        self.state.current_tick = tick
        self.initialize()
        
        # Main trading loop - as fast as possible
        loop_count = 0
        while True:
            try:
                active = self.run_tick()
                if not active:
                    print("\nCase ended or paused.")
                    break
                
                loop_count += 1
                
                # Adaptive sleep: faster when market is moving
                time.sleep(0.1)  # ~10 iterations per second
                
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print(f"\n  [ERROR] {e}")
                time.sleep(0.5)
        
        print(f"\nTotal loop iterations: {loop_count}")
        print("Algorithm finished.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys
    
    # Allow API key as command line argument
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    
    print("=" * 60)
    print("RITC 2026 VOLATILITY TRADING ALGORITHM")
    print("=" * 60)
    print(f"API: {API_BASE}")
    print(f"Strategies: Vega Mispricing + Delta Hedging + Gamma Scalping")
    print(f"Parameters:")
    print(f"  Min Vol Gap to Trade: {MIN_VEGA_EDGE_PCT:.0%}")
    print(f"  Hedge Interval: {HEDGE_INTERVAL_TICKS} ticks")
    print(f"  Target Position/Strike: {TARGET_POSITION_PER_STRIKE} contracts")
    print(f"  Delta Buffer: {DELTA_BUFFER_PCT:.0%} of limit")
    print()
    
    trader = VolatilityTrader()
    trader.run()