#!/usr/bin/env python3
"""
RITC 2026 Volatility Trading V4 - CAPTURE BOTH SIDES
=====================================================
DIAGNOSIS FROM TWO PRACTICE RUNS:

RUN 1 (V2-aggressive, FIX.pdf): $28.8K final, peaked at $130K
  - Made money early, bled it all back via churning and RTM commissions
  - RTM: -$2.9K realized, -$15.5K commissions, -$7.9K fines
  - 1.5M RTM shares traded = death by commission

RUN 2 (V1-original, FIX3.pdf): $41.8K final, dropped to -$30K first  
  - P&L went to -$30K in ticks 20-80 before recovering to +$55K by tick 160
  - Then slowly bled to $41.8K
  - Only 595K RTM shares ($6K comm) and 155 option trades ($10.8K comm)
  - RTM realized: -$87K (!!!) = catastrophic hedging losses
  - Big winners: C50 +$36.6K, C52 +$27K, C51 +$24.6K, C53 +$19.8K
  - All winners were CALLS bought low, sold high (underlying went UP)
  - All losers were PUTS bought at high IV that collapsed

THE CORE PROBLEM: Both algos get the initial week-start direction wrong.
  - At week start, IV hasn't adjusted to new RV yet
  - IV OVERSHOOTS before converging (market maker reprices slowly)
  - During this transition, the algo sells options (correct: IV > RV)
  - But selling options = short gamma = short the overshoot
  - If underlying MOVES (as it always does), short gamma LOSES

THE FIX - "CAPTURE BOTH SIDES":
  Phase 1 (Ticks 0-20 of each week): BALANCED STRADDLE SELLS
    - Sell EQUAL puts and calls at SAME strike (balanced straddle)
    - This captures vega edge while keeping delta near-zero
    - Net delta from options ≈ 0, so minimal RTM hedging needed
    - Key insight: sell C+P at SAME strike, not C at one strike + P at another
    
  Phase 2 (Ticks 20-50): HOLD + HEDGE MINIMALLY
    - IV converges toward RV, positions gain value
    - Only hedge when delta exceeds 80% of limit
    - Larger hedge increments, less frequent = fewer commissions
    
  Phase 3 (Ticks 50-70): TAKE PROFIT + RE-POSITION
    - Close winning positions (where IV has converged)  
    - Pre-position for next week using forecast
    - If forecast shows big vol change, build position in that direction

  Phase 4 (Ticks 70-75): FLATTEN FOR WEEK TRANSITION
    - Reduce positions ahead of new week's vol announcement
    - Fresh start each week

CRITICAL CHANGES FROM V1:
  1. BALANCED STRADDLES: sell C+P at same strike → near-zero delta
  2. MINIMIZE RTM: target <200K shares total (vs 595K or 1.5M)
  3. LARGER HEDGE SIZE: min 3000 shares (skip tiny hedges)
  4. BOTH SIDES: sell at overshoot, buy back at convergence
  5. WEEKLY RESET: flatten at week end, fresh start
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

API_BASE = "http://localhost:9995/v1"
API_KEY = "RYL2000"

# ----- Position Limits -----
GROSS_OPTION_LIMIT = 2400       # of 2500 max
NET_OPTION_LIMIT = 950          # of 1000 max
MAX_RTM_POSITION = 48000        # of 50000 max
MAX_CONTRACTS_PER_TRADE = 100   # max per order

# ----- Vega Trading -----
MIN_VEGA_EDGE_PCT = 0.03        # 3% IV-RV gap minimum
STRADDLE_QTY = 100              # contracts per leg of straddle
TARGET_PER_STRIKE = 100         # max position per strike

# ----- Delta Hedging (CONSERVATIVE to save commissions) -----
HEDGE_THRESHOLD_PCT = 0.75      # Only hedge above 75% of delta limit
EMERGENCY_HEDGE_PCT = 0.95      # Emergency hedge above 95%
MIN_HEDGE_SIZE = 3000           # Don't bother below this (saves $30 commission)
MAX_RTM_ORDERS_PER_CYCLE = 2    # Cap RTM orders per loop

# ----- Timing -----
TICKS_PER_WEEK = 75
TICKS_PER_DAY = 15
DAYS_PER_YEAR = 240
TOTAL_DAYS = 20
TOTAL_TICKS = 300
STRIKES = list(range(45, 55))
OPTION_MULTIPLIER = 100

# =============================================================================
# BLACK-SCHOLES
# =============================================================================

_INV_SQRT2PI = 1.0 / math.sqrt(2 * math.pi)

def _norm_cdf(x):
    if x < -8: return 0.0
    if x > 8: return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    p = _INV_SQRT2PI * math.exp(-0.5 * x * x)
    pp = p * t * (0.319381530 + t * (-0.356563782 + t * (
        1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - pp if x > 0 else pp

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) * _INV_SQRT2PI

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
    if T < 1e-8: return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
    return S * _norm_pdf(d1) * sqrtT * 0.01

def implied_vol_fast(price, S, K, T, r, is_call=True):
    if T < 1e-8: return None
    intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
    if price <= intrinsic + 0.003: return None
    sig = 0.30
    for _ in range(20):
        p = bs_price(S, K, T, r, sig, is_call)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
        vega = S * _norm_pdf(d1) * sqrtT
        if vega < 1e-12: break
        sig -= (p - price) / vega
        sig = max(0.005, min(sig, 2.0))
        if abs(p - price) < 1e-4: break
    return sig if 0.01 < sig < 2.0 else None

# =============================================================================
# API CLIENT
# =============================================================================

class RITClient:
    def __init__(self, api_key=API_KEY):
        self.session = requests.Session()
        self.session.headers.update({'X-API-Key': api_key})
        self.base = API_BASE
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            pool_connections=1, pool_maxsize=4))
    
    def get(self, endpoint, params=None):
        try:
            r = self.session.get(f"{self.base}{endpoint}", params=params, timeout=0.8)
            if r.ok: return r.json()
        except: pass
        return None
    
    def post(self, endpoint, params=None):
        try:
            r = self.session.post(f"{self.base}{endpoint}", params=params, timeout=0.8)
            if r.ok: return r.json()
        except: pass
        return None
    
    def delete(self, endpoint, params=None):
        try:
            r = self.session.delete(f"{self.base}{endpoint}", params=params, timeout=0.8)
            return r.ok
        except: return False
    
    def get_tick(self):
        data = self.get("/case")
        if data:
            return data.get('tick', 0), data.get('period', 1), data.get('status', 'STOPPED')
        return 0, 1, 'STOPPED'
    
    def get_securities(self):
        return self.get("/securities") or []
    
    def get_news(self, since=0):
        return self.get("/news", {'since': since}) or []
    
    def submit_order(self, ticker, side, qty, order_type='MARKET', price=None):
        params = {
            'ticker': ticker, 'type': order_type,
            'quantity': int(qty), 'action': side.upper(),
        }
        if price is not None:
            params['price'] = round(price, 2)
        return self.post("/orders", params)
    
    def cancel_all(self):
        return self.delete("/commands/cancel", {'all': 1})


# =============================================================================
# PORTFOLIO STATE
# =============================================================================

@dataclass
class PortfolioState:
    rtm_position: int = 0
    rtm_price: float = 50.0
    option_positions: Dict[str, int] = field(default_factory=dict)
    
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    
    current_tick: int = 0
    realized_vol: float = 0.30
    risk_free_rate: float = 0.0
    delta_limit: int = 10000
    penalty_pct: float = 0.03
    
    last_news_id: int = 0
    week_vols: Dict[int, float] = field(default_factory=dict)
    week_forecasts: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    
    reported_pnl: float = 0.0
    peak_pnl: float = 0.0
    
    _sec_cache: dict = field(default_factory=dict)
    
    def time_to_expiry(self):
        days = TOTAL_DAYS - (self.current_tick - 1) / TICKS_PER_DAY
        return max(days / DAYS_PER_YEAR, 1e-6)
    
    def current_week_num(self):
        return min((self.current_tick - 1) // TICKS_PER_WEEK + 1, 4)
    
    def ticks_into_week(self):
        return (self.current_tick - 1) % TICKS_PER_WEEK
    
    def get_realized_vol(self):
        wk = self.current_week_num()
        return self.week_vols.get(wk, self.realized_vol)
    
    def compute_greeks(self):
        T = self.time_to_expiry()
        S = self.rtm_price
        r = self.risk_free_rate
        rv = self.get_realized_vol()
        
        total_delta = float(self.rtm_position)
        total_gamma = 0.0
        total_vega = 0.0
        
        for ticker, qty in self.option_positions.items():
            if qty == 0: continue
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
# V4 TRADING ENGINE - BOTH SIDES CAPTURE
# =============================================================================

class VolatilityTraderV4:
    def __init__(self):
        self.client = RITClient()
        self.state = PortfolioState()
        self.last_hedge_tick = 0
        self.last_rtm_price = 50.0
        self.last_week = 0
        self.initialized = False
        
        # Track what we've done this week
        self.week_straddles_sold = {}    # strike -> qty sold this week
        self.week_positions_closed = {}  # strike -> True if closed this week
        
        # Stats
        self.rtm_shares_traded = 0
        self.option_trades = 0
        self.total_option_commission = 0
    
    # ==================== NEWS ====================
    
    def process_news(self):
        news_list = self.client.get_news(self.state.last_news_id)
        if not news_list: return
        
        for item in news_list:
            nid = item.get('news_id', 0)
            if nid <= self.state.last_news_id: continue
            self.state.last_news_id = nid
            
            headline = item.get('headline', '').lower()
            body = item.get('body', '')
            
            if 'delta limit' in headline:
                m = re.search(r'(\d+)', body)
                if m: self.state.delta_limit = int(m.group(1))
                m2 = re.search(r'(\d+)%', body)
                if m2: self.state.penalty_pct = float(m2.group(1)) / 100.0
                print(f"  [NEWS] Delta limit={self.state.delta_limit}, Pen={self.state.penalty_pct:.0%}")
            
            elif 'risk free rate' in headline:
                m = re.search(r'risk free rate is (\d+)%', body)
                if m: self.state.risk_free_rate = float(m.group(1)) / 100.0
                m = re.search(r'annualized realized volatility is (\d+)%', body)
                if m:
                    vol = float(m.group(1)) / 100.0
                    self.state.realized_vol = vol
                    print(f"  [NEWS] Initial vol={vol:.0%}")
            
            elif 'announcement' in headline:
                m_wk = re.search(r'(\d+)', item.get('headline', ''))
                m_vol = re.search(r'will be (\d+)%', body)
                if m_wk and m_vol:
                    wk = int(m_wk.group(1))
                    vol = float(m_vol.group(1)) / 100.0
                    self.state.week_vols[wk] = vol
                    print(f"  [NEWS] *** WEEK {wk} VOL = {vol:.0%} ***")
            
            elif 'news' in headline or 'forecast' in headline:
                m_wk = re.search(r'(\d+)', item.get('headline', ''))
                m_range = re.search(r'between (\d+)% and (\d+)%', body)
                if m_wk and m_range:
                    wk = int(m_wk.group(1))
                    lo = float(m_range.group(1)) / 100.0
                    hi = float(m_range.group(2)) / 100.0
                    self.state.week_forecasts[wk] = (lo, hi)
                    print(f"  [NEWS] Forecast wk{wk}: {lo:.0%}-{hi:.0%}")
    
    # ==================== MARKET DATA ====================
    
    def update_market_data(self):
        securities = self.client.get_securities()
        if not securities: return False
        
        sec_map = {}
        total_pnl = 0.0
        
        for sec in securities:
            ticker = sec.get('ticker', '')
            sec_map[ticker] = sec
            
            if ticker == 'RTM':
                self.state.rtm_position = int(sec.get('position', 0))
                bid = sec.get('bid', 0)
                ask = sec.get('ask', 0)
                last = sec.get('last', 50.0)
                self.state.rtm_price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
                total_pnl += sec.get('realized', 0) + sec.get('unrealized', 0)
            elif ticker.startswith('RTM1'):
                self.state.option_positions[ticker] = int(sec.get('position', 0))
                total_pnl += sec.get('realized', 0) + sec.get('unrealized', 0)
        
        self.state._sec_cache = sec_map
        self.state.reported_pnl = total_pnl
        self.state.peak_pnl = max(self.state.peak_pnl, total_pnl)
        self.state.compute_greeks()
        return True
    
    # ==================== STRATEGY 1: BALANCED STRADDLE SELLS ====================
    
    def find_best_straddle_strikes(self):
        """Find strikes where BOTH call and put are overpriced (IV > RV).
        Returns list of (strike, call_edge, put_edge, total_edge, call_iv, put_iv)."""
        
        S = self.state.rtm_price
        T = self.state.time_to_expiry()
        r = self.state.risk_free_rate
        rv = self.state.get_realized_vol()
        sec_map = self.state._sec_cache
        
        if T < 0.002: return []
        
        straddle_opps = []
        
        for K in STRIKES:
            call_ticker = f"RTM1C{K}"
            put_ticker = f"RTM1P{K}"
            
            call_sec = sec_map.get(call_ticker)
            put_sec = sec_map.get(put_ticker)
            if not call_sec or not put_sec: continue
            
            c_bid = call_sec.get('bid', 0)
            c_ask = call_sec.get('ask', 0)
            p_bid = put_sec.get('bid', 0)
            p_ask = put_sec.get('ask', 0)
            
            if c_bid <= 0 or c_ask <= 0 or p_bid <= 0 or p_ask <= 0:
                continue
            
            c_mid = (c_bid + c_ask) / 2.0
            p_mid = (p_bid + p_ask) / 2.0
            
            c_fair = bs_price(S, K, T, r, rv, True)
            p_fair = bs_price(S, K, T, r, rv, False)
            
            c_edge = c_mid - c_fair  # positive = overpriced (sell)
            p_edge = p_mid - p_fair
            
            c_iv = implied_vol_fast(c_mid, S, K, T, r, True)
            p_iv = implied_vol_fast(p_mid, S, K, T, r, False)
            
            if c_iv is None or p_iv is None: continue
            
            c_gap = c_iv - rv
            p_gap = p_iv - rv
            
            # BOTH sides must be overpriced for a straddle sell
            if c_gap > MIN_VEGA_EDGE_PCT and p_gap > MIN_VEGA_EDGE_PCT:
                total_edge = c_edge + p_edge  # total $ edge per share selling straddle
                
                # Compute net delta of balanced straddle (should be near 0)
                c_delta = bs_delta(S, K, T, r, rv, True)
                p_delta = bs_delta(S, K, T, r, rv, False)
                straddle_delta = c_delta + p_delta  # call delta + put delta (put is negative)
                
                straddle_opps.append({
                    'strike': K,
                    'c_edge': c_edge, 'p_edge': p_edge,
                    'total_edge': total_edge,
                    'c_iv': c_iv, 'p_iv': p_iv,
                    'c_gap': c_gap, 'p_gap': p_gap,
                    'straddle_delta': straddle_delta,
                    'c_bid': c_bid, 'p_bid': p_bid,
                })
        
        # Sort by total edge (best first)
        straddle_opps.sort(key=lambda x: x['total_edge'], reverse=True)
        return straddle_opps
    
    def execute_straddle_sells(self, opportunities):
        """Sell balanced straddles - equal calls and puts at same strike."""
        if not opportunities: return 0
        
        gross = self.state.gross_option_position()
        trades_done = 0
        
        for opp in opportunities:
            if trades_done >= 4: break  # Max 4 straddles per cycle (= 8 orders)
            
            K = opp['strike']
            
            # Skip if already sold straddles at this strike this week
            if self.week_straddles_sold.get(K, 0) >= TARGET_PER_STRIKE:
                continue
            
            call_ticker = f"RTM1C{K}"
            put_ticker = f"RTM1P{K}"
            
            c_pos = self.state.option_positions.get(call_ticker, 0)
            p_pos = self.state.option_positions.get(put_ticker, 0)
            
            # How many more can we sell?
            qty = min(STRADDLE_QTY, 
                      TARGET_PER_STRIKE - abs(c_pos),
                      TARGET_PER_STRIKE - abs(p_pos),
                      (GROSS_OPTION_LIMIT - gross) // 2)  # Need room for both legs
            
            if qty <= 0: continue
            qty = min(qty, MAX_CONTRACTS_PER_TRADE)
            
            # Sell call
            r1 = self.client.submit_order(call_ticker, 'SELL', qty, 'MARKET')
            # Sell put
            r2 = self.client.submit_order(put_ticker, 'SELL', qty, 'MARKET')
            
            if r1 and r2:
                self.state.option_positions[call_ticker] = c_pos - qty
                self.state.option_positions[put_ticker] = p_pos - qty
                gross += qty * 2
                trades_done += 1
                self.option_trades += 2
                self.total_option_commission += qty * 2
                
                self.week_straddles_sold[K] = self.week_straddles_sold.get(K, 0) + qty
                
                edge_total = opp['total_edge'] * qty * OPTION_MULTIPLIER
                print(f"  [STRADDLE SELL] K={K} x{qty}  "
                      f"C_IV={opp['c_iv']:.0%} P_IV={opp['p_iv']:.0%}  "
                      f"RV={self.state.get_realized_vol():.0%}  "
                      f"Δ={opp['straddle_delta']:+.3f}  "
                      f"edge=${edge_total:+,.0f}")
            elif r1:
                # Only call went through - immediately buy it back or accept
                self.state.option_positions[call_ticker] = c_pos - qty
                gross += qty
                self.option_trades += 1
                print(f"  [WARN] Only call leg filled for K={K}")
            elif r2:
                self.state.option_positions[put_ticker] = p_pos - qty
                gross += qty
                self.option_trades += 1
                print(f"  [WARN] Only put leg filled for K={K}")
        
        return trades_done
    
    # ==================== STRATEGY 2: SINGLE-LEG VEGA (when straddle not available) ====================
    
    def find_single_leg_opportunities(self):
        """Find individual options that are mispriced (when balanced straddle isn't possible)."""
        S = self.state.rtm_price
        T = self.state.time_to_expiry()
        r = self.state.risk_free_rate
        rv = self.state.get_realized_vol()
        sec_map = self.state._sec_cache
        
        if T < 0.002: return []
        
        opps = []
        for K in STRIKES:
            for is_call in [True, False]:
                ticker = f"RTM1{'C' if is_call else 'P'}{K}"
                sec = sec_map.get(ticker)
                if not sec: continue
                
                bid = sec.get('bid', 0)
                ask = sec.get('ask', 0)
                if bid <= 0 or ask <= 0: continue
                
                mid = (bid + ask) / 2.0
                spread = ask - bid
                fair = bs_price(S, K, T, r, rv, is_call)
                edge = mid - fair
                
                iv = implied_vol_fast(mid, S, K, T, r, is_call)
                if iv is None: continue
                
                vol_gap = iv - rv
                if abs(vol_gap) < MIN_VEGA_EDGE_PCT: continue
                if abs(edge) < 0.05: continue  # Min $0.05 edge
                if spread > abs(edge) * 0.7: continue  # Spread too wide
                
                delta = bs_delta(S, K, T, r, rv, is_call)
                
                opps.append({
                    'ticker': ticker, 'strike': K, 'is_call': is_call,
                    'bid': bid, 'ask': ask, 'mid': mid, 'fair': fair,
                    'edge': edge, 'iv': iv, 'vol_gap': vol_gap,
                    'delta': delta,
                    'action': 'SELL' if vol_gap > 0 else 'BUY',
                })
        
        opps.sort(key=lambda x: abs(x['edge']), reverse=True)
        return opps
    
    def execute_single_leg(self, opportunities):
        """Execute single-leg trades, preferring delta-neutral pairs."""
        if not opportunities: return 0
        
        gross = self.state.gross_option_position()
        trades_done = 0
        
        for opp in opportunities:
            if trades_done >= 4: break
            
            ticker = opp['ticker']
            current_pos = self.state.option_positions.get(ticker, 0)
            
            if opp['action'] == 'SELL':
                qty = min(TARGET_PER_STRIKE + current_pos, MAX_CONTRACTS_PER_TRADE)
                if qty <= 0 or current_pos <= -TARGET_PER_STRIKE: continue
                if gross + qty > GROSS_OPTION_LIMIT: continue
                
                result = self.client.submit_order(ticker, 'SELL', qty, 'MARKET')
                if result:
                    self.state.option_positions[ticker] = current_pos - qty
                    gross += qty
                    trades_done += 1
                    self.option_trades += 1
                    self.total_option_commission += qty
                    print(f"  [SELL] {qty} {ticker} IV={opp['iv']:.0%} "
                          f"gap={opp['vol_gap']:+.0%} edge=${opp['edge']:.3f}")
            else:
                qty = min(TARGET_PER_STRIKE - current_pos, MAX_CONTRACTS_PER_TRADE)
                if qty <= 0 or current_pos >= TARGET_PER_STRIKE: continue
                if gross + qty > GROSS_OPTION_LIMIT: continue
                
                result = self.client.submit_order(ticker, 'BUY', qty, 'MARKET')
                if result:
                    self.state.option_positions[ticker] = current_pos + qty
                    gross += qty
                    trades_done += 1
                    self.option_trades += 1
                    self.total_option_commission += qty
                    print(f"  [BUY]  {qty} {ticker} IV={opp['iv']:.0%} "
                          f"gap={opp['vol_gap']:+.0%} edge=${opp['edge']:.3f}")
        
        return trades_done
    
    # ==================== STRATEGY 3: TAKE PROFIT (Buy back straddles) ====================
    
    def take_profit_on_convergence(self):
        """When IV has converged to RV, buy back short options to lock in profit."""
        S = self.state.rtm_price
        T = self.state.time_to_expiry()
        r = self.state.risk_free_rate
        rv = self.state.get_realized_vol()
        sec_map = self.state._sec_cache
        
        trades_done = 0
        
        for ticker, qty in list(self.state.option_positions.items()):
            if qty >= 0: continue  # Only close SHORT positions (buy back)
            if trades_done >= 4: break
            
            sec = sec_map.get(ticker)
            if not sec: continue
            
            bid = sec.get('bid', 0)
            ask = sec.get('ask', 0)
            if ask <= 0: continue
            
            mid = (bid + ask) / 2.0
            is_call = 'C' in ticker[4:]
            strike = int(ticker[-2:])
            
            iv = implied_vol_fast(mid, S, strike, T, r, is_call)
            if iv is None: continue
            
            vol_gap = iv - rv
            
            # Take profit when IV has converged (gap < 2%) or reversed
            if abs(vol_gap) < 0.02 or vol_gap < -0.01:
                close_qty = min(abs(qty), MAX_CONTRACTS_PER_TRADE)
                result = self.client.submit_order(ticker, 'BUY', close_qty, 'MARKET')
                if result:
                    self.state.option_positions[ticker] = qty + close_qty
                    trades_done += 1
                    self.option_trades += 1
                    self.total_option_commission += close_qty
                    K = int(ticker[-2:])
                    self.week_positions_closed[K] = True
                    print(f"  [TAKE PROFIT] Buy back {close_qty} {ticker} "
                          f"(IV converged: {iv:.0%} vs RV {rv:.0%})")
        
        return trades_done
    
    # ==================== DELTA HEDGING (minimal) ====================
    
    def hedge_delta(self, force=False):
        """Conservative delta hedging - minimize RTM trades."""
        self.state.compute_greeks()
        delta = self.state.portfolio_delta
        delta_limit = self.state.delta_limit
        abs_delta = abs(delta)
        delta_ratio = abs_delta / max(delta_limit, 1)
        
        if not force:
            if delta_ratio < HEDGE_THRESHOLD_PCT:
                return  # Don't hedge below threshold
            if abs_delta < MIN_HEDGE_SIZE and delta_ratio < EMERGENCY_HEDGE_PCT:
                return
        
        hedge_qty = -int(round(delta))
        
        if abs(hedge_qty) < MIN_HEDGE_SIZE and not force:
            return
        
        # Clip to RTM limits
        new_pos = self.state.rtm_position + hedge_qty
        if new_pos > MAX_RTM_POSITION:
            hedge_qty = MAX_RTM_POSITION - self.state.rtm_position
        elif new_pos < -MAX_RTM_POSITION:
            hedge_qty = -MAX_RTM_POSITION - self.state.rtm_position
        
        if abs(hedge_qty) < 500:
            return
        
        remaining = abs(hedge_qty)
        side = 'BUY' if hedge_qty > 0 else 'SELL'
        chunks = 0
        
        while remaining > 0 and chunks < MAX_RTM_ORDERS_PER_CYCLE:
            chunk = min(remaining, 10000)
            result = self.client.submit_order('RTM', side, chunk, 'MARKET')
            if result:
                adj = chunk if side == 'BUY' else -chunk
                self.state.rtm_position += adj
                remaining -= chunk
                self.rtm_shares_traded += chunk
                chunks += 1
            else:
                break
        
        old_delta = delta
        self.state.compute_greeks()
        self.last_hedge_tick = self.state.current_tick
        
        if chunks > 0:
            comm = (abs(hedge_qty) - remaining) * 0.01
            print(f"  [HEDGE] Δ: {old_delta:+,.0f}->{self.state.portfolio_delta:+,.0f} "
                  f"RTM={self.state.rtm_position:+,d} cost=${comm:.0f}")
    
    # ==================== CLOSE-OUT ====================
    
    def close_positions(self):
        for ticker, qty in list(self.state.option_positions.items()):
            if qty == 0: continue
            close_qty = min(abs(qty), 100)
            side = 'SELL' if qty > 0 else 'BUY'
            self.client.submit_order(ticker, side, close_qty, 'MARKET')
        
        if abs(self.state.rtm_position) > 500:
            side = 'SELL' if self.state.rtm_position > 0 else 'BUY'
            qty = min(abs(self.state.rtm_position), 10000)
            self.client.submit_order('RTM', side, qty, 'MARKET')
    
    # ==================== MAIN LOOP ====================
    
    def initialize(self):
        print("\n" + "=" * 70)
        print("RITC 2026 VOL TRADER V4 - BALANCED STRADDLE STRATEGY")
        print("=" * 70)
        
        self.process_news()
        self.update_market_data()
        
        print(f"  RTM: ${self.state.rtm_price:.2f}")
        print(f"  Vol: {self.state.realized_vol:.0%}")
        print(f"  Delta Limit: {self.state.delta_limit}")
        print(f"  Strategy: Balanced straddle sells + convergence buyback")
        print(f"  Goal: <300K RTM shares, structured weekly phases")
        
        self.last_rtm_price = self.state.rtm_price
        self.initialized = True
    
    def run_tick(self):
        tick, period, status = self.client.get_tick()
        if status != 'ACTIVE':
            return False
        
        self.state.current_tick = tick
        if not self.update_market_data():
            return True
        
        self.process_news()
        
        wk = self.state.current_week_num()
        tiw = self.state.ticks_into_week()
        
        # Week change → reset tracking
        if wk != self.last_week:
            self.week_straddles_sold.clear()
            self.week_positions_closed.clear()
            self.last_week = wk
            print(f"\n  {'='*60}")
            print(f"  WEEK {wk} | Vol={self.state.get_realized_vol():.0%} | tick {tick}")
            print(f"  {'='*60}")
        
        delta_ratio = abs(self.state.portfolio_delta) / max(self.state.delta_limit, 1)
        
        # ===== P0: EMERGENCY HEDGE =====
        if delta_ratio > EMERGENCY_HEDGE_PCT:
            self.hedge_delta(force=True)
        
        # ===== PHASE 1: WEEK START (ticks 0-20) - SELL BALANCED STRADDLES =====
        if tiw <= 20 and tick < 275:
            if tick % 3 == 0:  # Every 3 ticks
                straddle_opps = self.find_best_straddle_strikes()
                if straddle_opps:
                    self.execute_straddle_sells(straddle_opps)
                else:
                    # Fallback to single-leg if no balanced straddle available
                    single_opps = self.find_single_leg_opportunities()
                    if single_opps:
                        self.execute_single_leg(single_opps[:4])
        
        # ===== PHASE 2: MID-WEEK (ticks 20-50) - HOLD + MINIMAL HEDGE =====
        elif tiw <= 50 and tick < 275:
            # Check for additional opportunities every 8 ticks
            if tick % 8 == 0:
                straddle_opps = self.find_best_straddle_strikes()
                if straddle_opps and straddle_opps[0]['total_edge'] > 0.15:
                    self.execute_straddle_sells(straddle_opps[:2])
            
            # Take profit on converging positions every 5 ticks
            if tick % 5 == 0:
                self.take_profit_on_convergence()
        
        # ===== PHASE 3: LATE WEEK (ticks 50-70) - TAKE PROFIT + PRE-POSITION =====
        elif tiw <= 70 and tick < 275:
            if tick % 4 == 0:
                self.take_profit_on_convergence()
            
            # Single-leg opportunities (smaller size)
            if tick % 6 == 0:
                single_opps = self.find_single_leg_opportunities()
                if single_opps:
                    self.execute_single_leg(single_opps[:2])
        
        # ===== PHASE 4: WEEK END (ticks 70-75) - FLATTEN =====
        elif tiw > 70 and tick < 275:
            self.close_positions()
        
        # ===== DELTA HEDGE (every 5 ticks, not every tick) =====
        if tick - self.last_hedge_tick >= 5 or delta_ratio > HEDGE_THRESHOLD_PCT:
            self.hedge_delta()
        
        # ===== CLOSE-OUT FINAL =====
        if tick > 280:
            self.close_positions()
        elif tick > 270:
            self.close_positions()
        
        # ===== STATUS =====
        if tick % 25 == 0:
            rv = self.state.get_realized_vol()
            pnl = self.state.reported_pnl
            rtm_comm = self.rtm_shares_traded * 0.01
            opt_comm = self.total_option_commission
            
            print(f"\n  ╔═ t={tick:3d} Wk{wk} Phase{1 if tiw<=20 else 2 if tiw<=50 else 3 if tiw<=70 else 4} ═══════════════════════════════╗")
            print(f"  ║ RTM=${self.state.rtm_price:.2f}  RV={rv:.0%}  P&L=${pnl:+,.0f} (peak ${self.state.peak_pnl:,.0f})")
            print(f"  ║ Δ={self.state.portfolio_delta:+,.0f}/{self.state.delta_limit} ({delta_ratio:.0%})")
            print(f"  ║ γ={self.state.portfolio_gamma:+,.0f}  ν={self.state.portfolio_vega:+,.0f}")
            print(f"  ║ Opts: gross={self.state.gross_option_position()}/2500  RTM={self.state.rtm_position:+,d}")
            print(f"  ║ Costs: RTM comm=${rtm_comm:,.0f} ({self.rtm_shares_traded:,d}sh)  Opt comm=${opt_comm:,.0f}")
            print(f"  ║ Trades: {self.option_trades} opt orders")
            print(f"  ╚═══════════════════════════════════════════════════════╝\n")
        
        self.last_rtm_price = self.state.rtm_price
        return True
    
    def run(self):
        print("\nWaiting for case to start...")
        while True:
            tick, period, status = self.client.get_tick()
            if status == 'ACTIVE': break
            time.sleep(0.3)
        
        print(f"Case ACTIVE! Tick={tick}")
        self.state.current_tick = tick
        self.initialize()
        
        while True:
            try:
                if not self.run_tick():
                    break
                time.sleep(0.1)  # 10 loops/sec
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"  [ERROR] {e}")
                time.sleep(0.5)
        
        rtm_comm = self.rtm_shares_traded * 0.01
        print(f"\n{'='*70}")
        print(f"FINAL: P&L=${self.state.reported_pnl:+,.0f}  Peak=${self.state.peak_pnl:,.0f}")
        print(f"RTM shares={self.rtm_shares_traded:,d} (${rtm_comm:,.0f} comm)")
        print(f"Option trades={self.option_trades} (${self.total_option_commission:,.0f} comm)")
        print(f"{'='*70}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    
    print("=" * 70)
    print("RITC 2026 VOLATILITY V4 - BALANCED STRADDLE BOTH-SIDES CAPTURE")
    print("=" * 70)
    print()
    print("STRATEGY: Sell balanced straddles (C+P same strike) when IV>RV")
    print("  → Near-zero delta from options (no directional bet)")
    print("  → Minimal RTM hedging needed (huge commission savings)")  
    print("  → Buy back when IV converges (capture both sides)")
    print("  → Weekly phase structure: sell → hold → profit → flatten")
    print()
    
    trader = VolatilityTraderV4()
    trader.run()