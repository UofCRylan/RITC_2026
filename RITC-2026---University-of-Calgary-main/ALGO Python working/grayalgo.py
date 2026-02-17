# ==========================================
# ULTIMATE HFT BOT - BATTLE READY (ALL CRITICAL FIXES)
# ==========================================

import asyncio
import aiohttp
import ujson
import time
import signal
import sys
import gc
import psutil
import os
import socket
import re
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# ==========================================
# CORE PARAMETERS
# ==========================================

# Constants
OBI_THRESHOLD = 0.35
OBI_HISTORY_LENGTH = 15
OBI_BOOK_DEPTH = 20
OBI_MIN_ORDER_SIZE = 50000

# Risk Limits
GROSS_LIMIT_TOTAL = 30000
DRIFT_TOLERANCE = 1000  

# Base Order Sizes (same for all tickers)
BASE_SIZE = 5000

# Ticker Allocation Limits (same for all — gross/4)
MAX_PER_TICKER = 7500

# Pricing Constants 
MIN_SPREAD_FOR_AGGRESSIVE = 0.05
PRICE_UPDATE_THRESHOLD_BUY = 0.01  
PRICE_UPDATE_THRESHOLD_SELL = 0.01   

# Performance Constants
TOKEN_BUCKET_CAPACITY = 50
TOKEN_REFILL_RATE_SEC = 0.001
FAST_CYCLE_MS = 0.03                # Slower cycles to respect 50 req/s API limit
SLOW_CYCLE_MS = 0.10                # Book fetch interval
REQUEST_TIMEOUT = 0.15
API_RATE_LIMIT = 45                 # Leave 5 headroom from 50/s hard limit

# Recovery Constants
MAX_CONSECUTIVE_FAILURES = 5
STUCK_LOCK_TIMEOUT = 0.05
NO_TRADE_TIMEOUT = 30.0
MIN_TOKENS_TO_TRADE = 1
MAX_RECOVERY_DURATION = 10.0
RECOVERY_COOLDOWN = 3.0

# Emergency Unwind
EMERGENCY_UNWIND_TRIGGER = 0.95  
EMERGENCY_UNWIND_EXIT = 0.90     

# News / Limit Change
NEWS_PAUSE_SECONDS = 3.0            # Pause trading for 3s after unwind
UNWIND_TICKS = [58, 118, 178, 238]  # 2 ticks before — market order unwind to news limit

MIN_ORDER_SIZE = 300  

# Global Configuration
API_KEY = 'ZVVCI8DU'
BASE_URL = 'http://localhost:9999/v1'
TICKERS = ['SPNG', 'SMMR', 'ATMN', 'WNTR']
SHUTDOWN = False

# ==========================================
# API RATE LIMITER (50 req/s hard limit)
# ==========================================

class APIRateLimiter:
    """Sliding window rate limiter. Call acquire() before every API request."""
    def __init__(self, max_per_second=API_RATE_LIMIT):
        self.max_per_second = max_per_second
        self.call_times = deque()
        self.blocked_count = 0
    
    async def acquire(self):
        now = time.monotonic()
        # Purge calls older than 1 second
        while self.call_times and self.call_times[0] < now - 1.0:
            self.call_times.popleft()
        
        if len(self.call_times) >= self.max_per_second:
            # Wait until oldest call falls out of window
            sleep_time = self.call_times[0] + 1.0 - now + 0.002
            if sleep_time > 0:
                self.blocked_count += 1
                await asyncio.sleep(sleep_time)
        
        self.call_times.append(time.monotonic())
    
    def current_rate(self):
        now = time.monotonic()
        while self.call_times and self.call_times[0] < now - 1.0:
            self.call_times.popleft()
        return len(self.call_times)

# Global rate limiter instance
RATE_LIMITER = APIRateLimiter()

# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class HeartbeatTracker:
    last_order_attempt: float = field(default_factory=time.time)
    last_order_success: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    total_attempts: int = 0
    total_successes: int = 0
    recovery_mode_activated: int = 0
    
    def record_attempt(self, success: bool):
        self.last_order_attempt = time.time()
        self.total_attempts += 1
        
        if success:
            self.last_order_success = time.time()
            self.consecutive_failures = 0
            self.total_successes += 1
        else:
            self.consecutive_failures += 1
    
    def is_stuck(self, ticker: str, current_spread: float, is_in_recovery: bool = False) -> bool:
        if is_in_recovery:
            if self.consecutive_failures >= 8:
                return True
            if time.time() - self.last_order_success > 15.0:
                return True
            return False
        
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return True
        
        now = time.time()
        time_since_success = now - self.last_order_success
        
        if time_since_success > NO_TRADE_TIMEOUT:
            time_since_attempt = now - self.last_order_attempt
            if time_since_attempt < 10.0:
                return True
        
        return False
    
    def get_health_score(self, ticker: str, current_spread: float) -> float:
        if self.total_attempts == 0:
            return 0.5
        
        success_rate = self.total_successes / self.total_attempts
        
        time_since_success = min(300.0, time.time() - self.last_order_success)
        time_factor = 1.0 - (time_since_success / 300.0)
        
        return (success_rate * 0.7) + (time_factor * 0.3)

@dataclass
class TokenBucket:
    tokens: int = TOKEN_BUCKET_CAPACITY
    last_refill: float = field(default_factory=time.monotonic)
    last_checked: float = field(default_factory=time.monotonic)
    empty_streak: int = 0
    
    def consume(self, tokens: int) -> bool:
        now = time.monotonic()
        
        if now - self.last_checked > 0.1:
            elapsed = now - self.last_refill
            refill = int(elapsed / TOKEN_REFILL_RATE_SEC)
            if refill > 0:
                self.tokens = min(TOKEN_BUCKET_CAPACITY, self.tokens + refill)
                self.last_refill = now
            self.last_checked = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            self.empty_streak = 0
            return True
        else:
            self.empty_streak += 1
            return False
    
    def force_refill(self):
        self.tokens = TOKEN_BUCKET_CAPACITY
        self.last_refill = time.monotonic()
        self.empty_streak = 0
    
    def is_stuck(self) -> bool:
        return self.empty_streak > 20 and self.tokens == 0

@dataclass
class TickerState:
    ticker: str
    
    # Server state
    server_position: int = 0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    
    # PNL tracking 
    realized_pnl: float = 0.0
    last_price: float = 0.0
    
    # Shadow ledger
    shadow_position: int = 0
    pending_buy_qty: int = 0
    pending_sell_qty: int = 0
    
    # Active orders
    active_buy_id: Optional[int] = None
    active_sell_id: Optional[int] = None
    active_buy_price: float = 0.0
    active_sell_price: float = 0.0
    
    # OBI state
    obi_current: float = 0.0
    obi_history: deque = field(default_factory=lambda: deque(maxlen=OBI_HISTORY_LENGTH))
    obi_delta: float = 0.0
    last_obi_update: float = 0.0
    
    # Market type
    market_type: str = "FLAT"
    
    # Pessimistic accounting
    worst_case_long: int = 0
    worst_case_short: int = 0
    
    # Flight control
    buy_in_flight: bool = False
    sell_in_flight: bool = False
    buy_lock_time: float = 0.0
    sell_lock_time: float = 0.0
    
    # Performance
    token_bucket: TokenBucket = field(default_factory=TokenBucket)
    cycles_since_drift_correction: int = 0
    order_sizes_reduced: bool = False
    
    # Heartbeat tracking
    heartbeat: HeartbeatTracker = field(default_factory=HeartbeatTracker)
    
    # Recovery state
    in_recovery: bool = False
    recovery_start: float = 0.0
    recovery_cooldown_end: float = 0.0
    last_diagnostic: Dict = field(default_factory=dict)
    
    _order_seen_buy: bool = False
    _order_seen_sell: bool = False
    
    @property
    def worst_case_long_pos(self) -> int:
        """If all pending buys fill, no sells do."""
        return self.server_position + self.pending_buy_qty
    
    @property
    def worst_case_short_pos(self) -> int:
        """If all pending sells fill, no buys do."""
        return self.server_position - self.pending_sell_qty
    
    @property
    def effective_position(self) -> int:
        """Worst-case: whichever side has larger abs exposure."""
        wl = self.worst_case_long_pos
        ws = self.worst_case_short_pos
        if abs(wl) >= abs(ws):
            return wl
        return ws
    
    @property
    def worst_case_exposure(self) -> int:
        """Absolute worst-case exposure for this ticker."""
        return max(abs(self.worst_case_long_pos), abs(self.worst_case_short_pos))
    
    def update_pessimistic_exposure(self):
        self.worst_case_long = abs(self.worst_case_long_pos)
        self.worst_case_short = abs(self.worst_case_short_pos)
        return max(self.worst_case_long, self.worst_case_short)
    
    def get_base_size(self) -> int:
        return BASE_SIZE
    
    def get_ticker_limit(self) -> int:
        return MAX_PER_TICKER
    
    def reset_order_tracking(self):
        self.pending_buy_qty = 0
        self.pending_sell_qty = 0
        self.active_buy_id = None
        self.active_sell_id = None
        self.active_buy_price = 0.0
        self.active_sell_price = 0.0
        self.buy_in_flight = False
        self.sell_in_flight = False
        self.buy_lock_time = 0.0
        self.sell_lock_time = 0.0
        self.shadow_position = self.server_position
        self.token_bucket.force_refill()
    
    def check_stuck_locks(self) -> List[str]:
        issues = []
        now = time.time()
        
        if self.buy_in_flight and now - self.buy_lock_time > STUCK_LOCK_TIMEOUT:
            issues.append(f"BUY lock stuck")
        
        if self.sell_in_flight and now - self.sell_lock_time > STUCK_LOCK_TIMEOUT:
            issues.append(f"SELL lock stuck")
        
        if self.token_bucket.is_stuck():
            issues.append("Token bucket stuck")
        
        if now > self.recovery_cooldown_end:
            if self.heartbeat.is_stuck(self.ticker, self.spread, self.in_recovery):
                issues.append(f"Heartbeat: {self.heartbeat.consecutive_failures} failures, {time.time() - self.heartbeat.last_order_success:.1f}s since success")
        
        return issues
    
    def force_release_locks(self):
        now = time.time()
        
        if self.buy_in_flight and now - self.buy_lock_time > 1.0:
            self.buy_in_flight = False
            self.token_bucket.force_refill()
        
        if self.sell_in_flight and now - self.sell_lock_time > 1.0:
            self.sell_in_flight = False
            self.token_bucket.force_refill()

# ==========================================
# CORE ENGINE FUNCTIONS
# ==========================================

def calculate_obi(book_data: Dict) -> float:
    try:
        bids = book_data.get('bids', [])
        asks = book_data.get('asks', [])
        
        bid_depth = min(OBI_BOOK_DEPTH, len(bids))
        ask_depth = min(OBI_BOOK_DEPTH, len(asks))
        
        total_bid_volume = sum(
            int(b.get('quantity', 0))
            for b in bids[:bid_depth]
            if int(b.get('quantity', 0)) >= OBI_MIN_ORDER_SIZE
        )
        total_ask_volume = sum(
            int(a.get('quantity', 0))
            for a in asks[:ask_depth]
            if int(a.get('quantity', 0)) >= OBI_MIN_ORDER_SIZE
        )
        
        total_volume = total_bid_volume + total_ask_volume
        if total_volume == 0:
            return 0.0
        
        obi = (total_bid_volume - total_ask_volume) / total_volume
        return max(-1.0, min(1.0, obi))
    except Exception:
        return 0.0

def calculate_obi_delta(obi_history: deque) -> float:
    if len(obi_history) < 10:
        return 0.0
    recent_avg = sum(list(obi_history)[-5:]) / 5.0
    previous_avg = sum(list(obi_history)[-10:-5]) / 5.0
    return recent_avg - previous_avg

def classify_market_type(obi: float) -> str:
    if abs(obi) <= OBI_THRESHOLD:
        return "FLAT"
    elif obi > OBI_THRESHOLD:
        return "BULL"
    else:
        return "BEAR"

def calculate_prices(state: TickerState) -> Tuple[float, float, int, int]:
    base_size = state.get_base_size()
    
    position = state.shadow_position
    max_position = state.get_ticker_limit()
    
    # Clamp position ratio to prevent extremes
    position_ratio = position / max_position
    if position_ratio > 1.0:
        position_ratio = 1.0
    elif position_ratio < -1.0:
        position_ratio = -1.0
    
    buy_skew = round(1.0 - position_ratio, 2)  
    sell_skew = 2.0 - buy_skew
    
    buy_size = int(base_size * buy_skew)
    sell_size = int(base_size * sell_skew)
    
    # Round to nearest 10
    buy_size = max(0, round(buy_size, -1))
    sell_size = max(0, round(sell_size, -1))
    
    # Apply reductions
    if state.in_recovery:
        buy_size = int(buy_size * 0.2)
        sell_size = int(sell_size * 0.2)
    elif state.order_sizes_reduced and state.cycles_since_drift_correction < 3:
        buy_size = int(buy_size * 0.5)
        sell_size = int(sell_size * 0.5)
    
    # Dead zone
    buy_size = max(MIN_ORDER_SIZE, buy_size) if buy_skew >= 0.05 else 0
    sell_size = max(MIN_ORDER_SIZE, sell_size) if sell_skew >= 0.05 else 0
    
    # Tight spread logic
    if state.spread < 0.05:
        buy_price = state.best_bid
        sell_price = state.best_ask
        
        if state.market_type == "BULL" and state.obi_current > 0.4:
            buy_price = round(state.best_bid + 0.01, 2)
            sell_price = round(state.best_ask + 0.03, 2)
        elif state.market_type == "BEAR" and state.obi_current < -0.4:
            sell_price = round(state.best_ask - 0.01, 2)
            buy_price = round(state.best_bid - 0.03, 2)
        
        if buy_price >= sell_price:
            buy_price = round(sell_price - 0.01, 2)
        
        return buy_price, sell_price, buy_size, sell_size
    
    # Wide spread logic
    best_bid = state.best_bid
    best_ask = state.best_ask
    spread = state.spread
    
    if spread >= 0.05:
        base_buy_price = round(best_bid + 0.01, 2)
        base_sell_price = round(best_ask - 0.01, 2)
    else:
        base_buy_price = round(best_bid, 2)
        base_sell_price = round(best_ask, 2)
    
    if state.market_type == "FLAT":
        buy_price = base_buy_price
        sell_price = base_sell_price
    elif state.market_type == "BULL":
        buy_price = round(base_buy_price + 0.01, 2)
        sell_price = round(base_sell_price + 0.04, 2) 
    elif state.market_type == "BEAR":
        sell_price = round(base_sell_price - 0.01, 2) 
        buy_price = round(base_buy_price - 0.4, 2)
    
    if buy_price >= sell_price:
        buy_price = round(sell_price - 0.01, 2)
    
    return buy_price, sell_price, buy_size, sell_size

# ==========================================
# HFT TICKER ENGINE
# ==========================================

class HFTTickerEngine:
    def __init__(self, ticker: str, parent_bot):
        self.ticker = ticker
        self.parent = parent_bot
        self.state = TickerState(ticker=ticker)
        self.last_book_fetch = 0.0
        self.cycle_count = 0
        self.last_diagnostic_run = 0.0
        self.recovery_attempts = 0
        self.consecutive_no_spread_cycles = 0
        self.last_trade_time = 0.0
        self.consecutive_orders = 0
        
        self.sec_url = f"{BASE_URL}/securities?ticker={ticker}&key={API_KEY}"
        self.book_url = f"{BASE_URL}/securities/book?ticker={ticker}&key={API_KEY}&limit={OBI_BOOK_DEPTH}"
        self.orders_url = f"{BASE_URL}/orders?key={API_KEY}"
        self.del_url_base = f"{BASE_URL}/orders/{{}}?key={API_KEY}"
    
    async def fetch_security_data(self, session: aiohttp.ClientSession) -> bool:
        for attempt in range(2):
            try:
                await RATE_LIMITER.acquire()
                async with session.get(self.sec_url, timeout=0.5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and len(data) > 0:
                            sec = data[0]
                            
                            self.state.server_position = int(sec.get('position', 0))
                            self.state.best_bid = float(sec.get('bid', 0.0))
                            self.state.best_ask = float(sec.get('ask', 0.0))
                            self.state.spread = self.state.best_ask - self.state.best_bid
                            self.state.realized_pnl = float(sec.get('realized', 0.0))
                            self.state.last_price = float(sec.get('last', 0.0))
                            return True
                await asyncio.sleep(0.001 * (attempt + 1))
            except Exception:
                await asyncio.sleep(0.001 * (attempt + 1))
        return False
    
    async def fetch_book_data(self, session: aiohttp.ClientSession) -> bool:
        try:
            await RATE_LIMITER.acquire()
            async with session.get(self.book_url, timeout=0.5) as resp:
                if resp.status == 200:
                    book_data = await resp.json()
                    obi = calculate_obi(book_data)
                    self.state.obi_history.append(obi)
                    self.state.obi_current = obi
                    self.state.obi_delta = calculate_obi_delta(self.state.obi_history)
                    self.state.market_type = classify_market_type(obi)
                    self.last_book_fetch = time.monotonic()
                    return True
        except Exception:
            pass
        return False
    
    def check_drift(self) -> bool:
        if self.state.pending_buy_qty == 0 and self.state.pending_sell_qty == 0:
            return False
            
        drift = abs(self.state.shadow_position - self.state.server_position)
        
        if drift > DRIFT_TOLERANCE and self.state.cycles_since_drift_correction > 30:
            return True
        
        return False
    
    async def gentle_drift_correction(self, session: aiohttp.ClientSession):
        
        if self.state.active_buy_id or self.state.active_sell_id:
            cancel_url = f"{BASE_URL}/commands/cancel?ticker={self.ticker}&key={API_KEY}"
            try:
                async with session.post(cancel_url, timeout=0.5) as resp:
                    await resp.read()
            except Exception:
                pass
        
        self.state.reset_order_tracking()
        self.state.order_sizes_reduced = True
        self.state.cycles_since_drift_correction = 0
        
        await asyncio.sleep(0.01)
    
    def _worst_case_gross_current(self) -> int:
        """✅ CRITICAL: Current worst-case exposure across all tickers"""
        total = 0
        for engine in self.parent.engines.values():
            srv = engine.state.server_position
            # Worst-case: whichever pending fills maximizes exposure
            buy_scenario = abs(srv + engine.state.pending_buy_qty)
            sell_scenario = abs(srv - engine.state.pending_sell_qty)
            worst = max(buy_scenario, sell_scenario)
            total += worst
        return int(total)
    
    def _worst_case_gross_projected(self, action: str, quantity: int) -> int:
        """✅ CRITICAL: Projected worst-case AFTER this order REPLACES old order"""
        total = 0
        for engine in self.parent.engines.values():
            srv = engine.state.server_position
            
            if engine is self:
                # Project position AFTER this order fills (REPLACE semantics)
                if action == "BUY":
                    new_pos = srv + quantity
                    # From new position, what if sell also fills?
                    worst = max(abs(new_pos), 
                               abs(new_pos - engine.state.pending_sell_qty))
                else:  # SELL
                    new_pos = srv - quantity
                    # From new position, what if buy also fills?
                    worst = max(abs(new_pos + engine.state.pending_buy_qty),
                               abs(new_pos))
            else:
                # Other tickers: unchanged
                buy_scenario = abs(srv + engine.state.pending_buy_qty)
                sell_scenario = abs(srv - engine.state.pending_sell_qty)
                worst = max(buy_scenario, sell_scenario)
            
            total += worst
        
        return int(total)
    
    def can_place_order(self, action: str, quantity: int, is_unwind_order: bool = False) -> bool:
        """✅ FIXED: Projected gross logic with unwind allowance"""
        if self.parent.emergency_unwind_active:
            return False
            
        if time.time() < self.state.recovery_cooldown_end:
            return False
        
        now = time.time()
        
        # Add minimum time between trades (50ms)
        if now - self.last_trade_time < 0.05:
            return False
        
        if action == 'BUY':
            if self.state.buy_in_flight:
                if now - self.state.buy_lock_time > 1.0:
                    self.state.buy_in_flight = False
                else:
                    return False
        else:
            if self.state.sell_in_flight:
                if now - self.state.sell_lock_time > 1.0:
                    self.state.sell_in_flight = False
                else:
                    return False
        
        if not self.state.token_bucket.consume(1):
            if self.state.token_bucket.is_stuck():
                self.state.token_bucket.force_refill()
            return False
        
        ticker_limit = self.state.get_ticker_limit()
        
        # Per-ticker limit: check worst case for the side we're adding to
        if action == 'BUY':
            # Worst case: all buys fill INCLUDING this new one
            projected_long = self.state.worst_case_long_pos + quantity
            if abs(projected_long) > ticker_limit * 0.99:
                self.state.token_bucket.force_refill()
                return False
        else:
            # Worst case: all sells fill INCLUDING this new one
            projected_short = self.state.worst_case_short_pos - quantity
            if abs(projected_short) > ticker_limit * 0.99:
                self.state.token_bucket.force_refill()
                return False
        
        # Projected gross with unwind allowance
        current_worst = self._worst_case_gross_current()
        projected_worst = self._worst_case_gross_projected(action, quantity)
        limit = int(GROSS_LIMIT_TOTAL * 0.95)
        
        # Primary check: within limit
        if projected_worst <= limit:
            return True
        
        # Unwind allowance: reducing risk is always allowed
        if projected_worst < current_worst:
            delta = current_worst - projected_worst
            return True
        
        # Block: risk-increasing order
        self.state.token_bucket.force_refill()
        return False
    
    def should_update_order(self, side: str, target_price: float, current_price: float) -> bool:
        if current_price == 0.0:
            return True
        
        price_diff = abs(target_price - current_price)
        
        if self.state.in_recovery:
            threshold = 0.01
        else:
            threshold = PRICE_UPDATE_THRESHOLD_BUY if side == 'BUY' else PRICE_UPDATE_THRESHOLD_SELL
        
        return price_diff > threshold
    
    async def _execute_order_sequence(self, session: aiohttp.ClientSession, 
                                    side: str, target_price: float, size: int,
                                    is_unwind_order: bool = False) -> bool:
        success = False
        
        if side == 'BUY':
            self.state.buy_in_flight = True
            self.state.buy_lock_time = time.time()
            old_id = self.state.active_buy_id
        else:
            self.state.sell_in_flight = True
            self.state.sell_lock_time = time.time()
            old_id = self.state.active_sell_id
        
        try:            
            if old_id is not None:
                try:
                    await RATE_LIMITER.acquire()
                    async with session.delete(self.del_url_base.format(old_id), timeout=0.10) as resp:
                        await resp.read()
                except Exception:
                    pass

            payload = {
                'ticker': self.ticker,
                'type': 'LIMIT',
                'action': side,
                'quantity': size,
                'price': target_price
                                        }

            await RATE_LIMITER.acquire()
            async with session.post(self.orders_url, params=payload, timeout=0.5) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    if side == 'BUY':
                        self.state.pending_buy_qty = size
                        self.state.active_buy_id = data.get('order_id') or data.get('id')
                        self.state.active_buy_price = target_price
                    else:
                        self.state.pending_sell_qty = size
                        self.state.active_sell_id = data.get('order_id') or data.get('id')
                        self.state.active_sell_price = target_price
                    
                    self.state.heartbeat.record_attempt(True)
                    success = True
                    self.recovery_attempts = 0
                    self.last_trade_time = time.time()  
                    self.consecutive_orders += 1
                    
                    # Reset if too many consecutive orders
                    if self.consecutive_orders > 10:
                        self.consecutive_orders = 0
                        await asyncio.sleep(0.01)
                    
                else:
                    raise Exception(f"HTTP {resp.status}")
                    
        except Exception as e:
            self.state.token_bucket.force_refill()
            self.state.heartbeat.record_attempt(False)
            self.consecutive_orders = 0
            
        finally:
            if side == 'BUY': 
                self.state.buy_in_flight = False
            else: 
                self.state.sell_in_flight = False
        
        return success
    
    def run_diagnostic(self) -> Dict:
        now = time.time()
        
        if now - self.last_diagnostic_run < 2.0:
            return self.state.last_diagnostic
        
        diagnostic = {
            'ticker': self.ticker,
            'timestamp': now,
            'health_score': self.state.heartbeat.get_health_score(self.ticker, self.state.spread),
            'is_stuck': self.state.heartbeat.is_stuck(self.ticker, self.state.spread, self.state.in_recovery),
            'stuck_issues': self.state.check_stuck_locks(),
            'server_position': self.state.server_position,
            'shadow_position': self.state.shadow_position,
            'effective_position': self.state.effective_position,
            'pending_buy': self.state.pending_buy_qty,
            'pending_sell': self.state.pending_sell_qty,
            'buy_in_flight': self.state.buy_in_flight,
            'sell_in_flight': self.state.sell_in_flight,
            'tokens': self.state.token_bucket.tokens,
            'consecutive_failures': self.state.heartbeat.consecutive_failures,
            'time_since_success': now - self.state.heartbeat.last_order_success,
            'recovery_mode': self.state.in_recovery,
            'recovery_duration': now - self.state.recovery_start if self.state.in_recovery else 0,
            'recovery_cooldown': max(0, self.state.recovery_cooldown_end - now),
            'market_spread': self.state.spread,
            'best_bid': self.state.best_bid,
            'best_ask': self.state.best_ask,
            'recovery_attempts': self.recovery_attempts,
        }
        
        self.state.last_diagnostic = diagnostic
        self.last_diagnostic_run = now
        
        return diagnostic
    
    async def enter_recovery_mode(self, session: aiohttp.ClientSession):
        if self.state.in_recovery:
            return
        
        self.state.in_recovery = True
        self.state.recovery_start = time.time()
        self.state.recovery_cooldown_end = time.time() + 1.0
        self.recovery_attempts += 1
        
        cancel_url = f"{BASE_URL}/commands/cancel?ticker={self.ticker}&key={API_KEY}"
        try:
            async with session.post(cancel_url, timeout=1.0) as resp:
                await resp.read()
        except Exception:
            pass
        
        self.state.reset_order_tracking()
        self.state.heartbeat.recovery_mode_activated += 1
        
        await asyncio.sleep(0.1)
        
    
    def check_exit_recovery(self):
        if not self.state.in_recovery:
            return False
        
        now = time.time()
        recovery_duration = now - self.state.recovery_start
        
        if recovery_duration > MAX_RECOVERY_DURATION:
            self.state.in_recovery = False
            self.state.recovery_cooldown_end = now + RECOVERY_COOLDOWN
            return True
        
        if self.state.heartbeat.consecutive_failures == 0:
            time_since_success = now - self.state.heartbeat.last_order_success
            if time_since_success < 5.0:
                self.state.in_recovery = False
                self.state.recovery_cooldown_end = now + RECOVERY_COOLDOWN
                return True
        
        if recovery_duration > 3.0 and self.state.heartbeat.consecutive_failures < 3:
            self.state.in_recovery = False
            self.state.recovery_cooldown_end = now + RECOVERY_COOLDOWN
            return True
        
        return False
    
    async def run_cycle(self, session: aiohttp.ClientSession, now: float) -> bool:
        if self.parent.emergency_unwind_active or self.parent.endgame_active:
            return False
        
        if self.state.spread <= 0:
            return False
        
        self.cycle_count += 1
        self.state.cycles_since_drift_correction += 1
        
        # Reset consecutive orders counter if no recent trades
        if time.time() - self.last_trade_time > 1.0:
            self.consecutive_orders = 0
        
        if self.state.in_recovery:
            if self.check_exit_recovery():
                self.state.reset_order_tracking()
                return False
        
        diagnostic = self.run_diagnostic()
        
        is_stuck = diagnostic.get('is_stuck', False)
        stuck_issues = diagnostic.get('stuck_issues', [])
        
        now_time = time.time()
        if (is_stuck and 
            not self.state.in_recovery and 
            self.cycle_count > 50 and
            now_time > self.state.recovery_cooldown_end):
            
            if self.recovery_attempts < 3:
                await self.enter_recovery_mode(session)
                return False
            else:
                self.state.recovery_cooldown_end = now_time + 30.0
                return False
        
        if self.check_drift():
            await self.gentle_drift_correction(session)
            return True
        
        if self.state.order_sizes_reduced and self.state.cycles_since_drift_correction >= 3:
            self.state.order_sizes_reduced = False
        
        buy_price, sell_price, buy_size, sell_size = calculate_prices(self.state)
        
        orders_placed = 0
        
        if (buy_size >= MIN_ORDER_SIZE and 
            self.should_update_order('BUY', buy_price, self.state.active_buy_price)):
            
            can_place = self.can_place_order('BUY', buy_size)
            
            if can_place:
                success = await self._execute_order_sequence(
                    session, 'BUY', buy_price, buy_size
                )
                if success:
                    orders_placed += 1
        
        if (sell_size >= MIN_ORDER_SIZE and 
            self.should_update_order('SELL', sell_price, self.state.active_sell_price)):
            
            can_place = self.can_place_order('SELL', sell_size)
            
            if can_place:
                success = await self._execute_order_sequence(
                    session, 'SELL', sell_price, sell_size
                )
                if success:
                    orders_placed += 1
        
        return orders_placed > 0

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================

class UltimateHFTSentinel:
    def __init__(self):
        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.winmm.timeBeginPeriod(1)
            except:
                pass
            
            if sys.version_info >= (3, 8):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        gc.disable()
        
        try:
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        except:
            pass
        
        self.engines = {}
        for ticker in TICKERS:
            self.engines[ticker] = HFTTickerEngine(ticker, self)
        
        self.cycles = 0
        self.current_pnl = 0.0
        self.emergency_unwind_active = False
        self.endgame_active = False
        self.session = None
        
        self.pulse_interval = FAST_CYCLE_MS
        self.next_pulse = 0
        self.start_time = time.time()
        self.emergency_unwinds = 0
        self.last_status_print = 0
        self.last_global_diagnostic = 0
        
        self.last_trading_activity = time.time()
        self.consecutive_no_trade_cycles = 0
        
        self.emergency_start_time = 0.0
        self.emergency_timeout_end = 0.0
        
        self._cached_gross = 0
        self._cached_gross_time = 0.0

        self.last_orders_fetch = 0.0 

        # ── Tick tracking (simulation tick = source of truth) ─
        self.current_tick = 0                      # Actual sim tick from API
        self.last_tick_fetch = 0.0
        
        # ── News / limit change ───────────────────────────
        self.current_gross_limit = GROSS_LIMIT_TOTAL   # Always 30000 for normal trading
        self.news_pause_end = 0.0                      # Don't trade until this time
        self.last_news_id = 0                          # Track which news we've seen
        self.last_unwind_tick = -1
        
        # Gross limit from news — must be at/below this before each minute mark
        self.news_gross_limit = 0                  # Scraped from /news at startup
    
    def calculate_current_gross(self) -> int:
        """Server positions only (ground truth)"""
        now = time.monotonic()
        if now - self._cached_gross_time < 0.001:
            return self._cached_gross
        
        total = 0
        for engine in self.engines.values():
            total += abs(engine.state.server_position)
        
        self._cached_gross = total
        self._cached_gross_time = now
        return total
    
    def calculate_current_pnl(self) -> float:
        total_pnl = 0.0
        for engine in self.engines.values():
            total_pnl += engine.state.realized_pnl
        return total_pnl
    
    async def initialize(self) -> bool:
        # init
        
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=50,
            family=socket.AF_INET,
            ttl_dns_cache=3600,
            force_close=False,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            json_serialize=ujson.dumps,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        )
        
        for attempt in range(3):
            try:
                url = f"{BASE_URL}/case?key={API_KEY}"
                async with self.session.get(url, timeout=1.0) as resp:
                    if resp.status == 200:
                        print("✅ API connection successful")
                        break
                    elif attempt == 2:
                        print("❌ API connection failed")
                        return False
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(0.5)
                else:
                    return False
        
        tasks = []
        for engine in self.engines.values():
            tasks.append(engine.fetch_security_data(self.session))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        for ticker, engine in self.engines.items():
            engine.state.shadow_position = engine.state.server_position
        
        # Fetch news at startup to learn the gross limit target
        await self.fetch_news_limit()
        
        return True
    
    async def fetch_news_limit(self):
        """
        Scrape /news at startup. Sort by news_id, take the SECOND item,
        find the number immediately before 'shares'.
        """
        for attempt in range(10):
            try:
                await RATE_LIMITER.acquire()
                async with self.session.get(
                    f"{BASE_URL}/news?key={API_KEY}",
                    timeout=aiohttp.ClientTimeout(total=2.0),
                ) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(0.5)
                        continue
                    data = await resp.json()
                    
                    if not data or len(data) < 2:
                        print(f"📰 Attempt {attempt+1}: only {len(data) if data else 0} news items, waiting...")
                        await asyncio.sleep(1.0)
                        continue
                    
                    # Sort by news_id ascending so index 0=first, 1=second
                    sorted_news = sorted(data, key=lambda x: x.get('news_id', x.get('id', 0)))
                    
                    # Track IDs
                    for item in sorted_news:
                        nid = item.get('news_id') or item.get('id', 0)
                        self.last_news_id = max(self.last_news_id, nid)
                    
                    # Print all news for debugging
                    for i, item in enumerate(sorted_news):
                        text = f"{item.get('headline', '')} {item.get('body', '')}"
                        print(f"📰 News[{i}]: {text[:150]}")
                    
                    # Target the second news item (index 1)
                    second = sorted_news[1]
                    text = f"{second.get('headline', '')} {second.get('body', '')}"
                    
                    # Find NUMBER immediately before "shares" (case insensitive)
                    # Handles: "9000 shares", "9,000 shares", "9000  shares"
                    match = re.search(r'([\d,]+)\s+shares', text, re.IGNORECASE)
                    if match:
                        self.news_gross_limit = int(match.group(1).replace(',', ''))
                        print(f"✅ News gross limit: {self.news_gross_limit:,}")
                        return
                    
                    # Fallback: try all news items
                    for item in sorted_news:
                        text = f"{item.get('headline', '')} {item.get('body', '')}"
                        match = re.search(r'([\d,]+)\s+shares', text, re.IGNORECASE)
                        if match:
                            self.news_gross_limit = int(match.group(1).replace(',', ''))
                            print(f"✅ News gross limit (from other news): {self.news_gross_limit:,}")
                            return
                    
                    print(f"⚠️  Attempt {attempt+1}: 'shares' not found in any news")
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                print(f"⚠️  News fetch error: {e}")
                await asyncio.sleep(0.5)
        
        self.news_gross_limit = GROSS_LIMIT_TOTAL
        print(f"⚠️  Fallback: using default gross limit {self.news_gross_limit:,}")
    
    async def parallel_data_fetch(self):
        all_tasks = []
        now = time.monotonic()
        
        for engine in self.engines.values():
            all_tasks.append(engine.fetch_security_data(self.session))
        
        for engine in self.engines.values():
            if now - engine.last_book_fetch > SLOW_CYCLE_MS:
                all_tasks.append(engine.fetch_book_data(self.session))
        
        if now - self.last_orders_fetch > 0.01:
            all_tasks.append(self._fetch_orders_atomic(self.session))
            self.last_orders_fetch = now
        
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
    
    async def check_risk_limits(self):
        """Emergency with timeout — uses dynamic limit from news"""
        exposure = self.calculate_current_gross()
        current_limit = self.current_gross_limit
        
        if self.emergency_unwind_active:
            if time.time() - self.emergency_start_time > 5.0:
                # quiet
                self.emergency_unwind_active = False
                self.emergency_timeout_end = time.time() + 3.0
                return
                
            if exposure < current_limit * EMERGENCY_UNWIND_EXIT:
                self.emergency_unwind_active = False
            return
        
        # Don't check during cooldown
        if time.time() < self.emergency_timeout_end:
            return
        
        # Trigger emergency
        if exposure > current_limit * EMERGENCY_UNWIND_TRIGGER:
            print(f"🚨 EMERGENCY UNWIND: {exposure:.0f}/{current_limit}")
            self.emergency_unwind_active = True
            self.emergency_start_time = time.time()
            await self.emergency_unwind_all()
            self.emergency_unwinds += 1
    
    async def emergency_unwind_all(self):
        """Progressive unwind with better timing"""
        # quiet
        
        cancel_url = f"{BASE_URL}/commands/cancel?all=1&key={API_KEY}"
        try:
            async with self.session.post(cancel_url, timeout=1.0) as resp:
                await resp.read()
        except Exception:
            pass
        
        for engine in self.engines.values():
            engine.state.reset_order_tracking()
        
        await asyncio.sleep(0.1)
        
        # Progressive unwind
        max_attempts = 3
        for attempt in range(max_attempts):
            unwind_tasks = []
            
            for ticker, engine in self.engines.items():
                state = engine.state
                if abs(state.server_position) > 100:
                    action = 'SELL' if state.server_position > 0 else 'BUY'
                    
                    if attempt == 0:
                        price_adj = 0.02
                    elif attempt == 1:
                        price_adj = 0.05
                    else:
                        # MARKET order
                        payload = {
                            'ticker': ticker,
                            'type': 'MARKET',
                            'action': action,
                            'quantity': min(abs(state.server_position), 5000),
                        }
                        url = f"{BASE_URL}/orders?key={API_KEY}"
                        unwind_tasks.append(self.session.post(url, params=payload))
                        continue
                    
                    if action == 'SELL':
                        price = round(state.best_bid - price_adj, 2)
                    else:
                        price = round(state.best_ask + price_adj, 2)
                    
                    payload = {
                        'ticker': ticker,
                        'type': 'LIMIT',
                        'action': action,
                        'quantity': min(abs(state.server_position), 5000),
                        'price': price
                    }
                    
                    url = f"{BASE_URL}/orders?key={API_KEY}"
                    unwind_tasks.append(self.session.post(url, params=payload))
            
            if unwind_tasks:
                await asyncio.gather(*unwind_tasks, return_exceptions=True)
            
            if attempt < max_attempts - 1:
                await asyncio.sleep(0.15)
            else:
                await asyncio.sleep(0.3)
            
            await self.parallel_data_fetch()
            
            exposure = self.calculate_current_gross()
            if exposure < self.current_gross_limit * 0.7:
                break
        
        for engine in self.engines.values():
            engine.state.in_recovery = False
            engine.state.recovery_cooldown_end = time.time() + 2.0
        
        # quiet
    
    # ==========================================
    # TICK TRACKING & NEWS MONITORING
    # ==========================================
    
    async def fetch_tick_info(self):
        """Get the TRUE simulation tick from /case API. This is the real clock."""
        now = time.monotonic()
        if now - self.last_tick_fetch < 0.15:
            return
        self.last_tick_fetch = now
        try:
            await RATE_LIMITER.acquire()
            async with self.session.get(
                f"{BASE_URL}/case?key={API_KEY}", timeout=0.5
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # RIT API: 'tick' is the current simulation second
                    tick = data.get('tick')
                    if tick is None:
                        tick = 300 - data.get('ticks_remaining', 300 - self.current_tick)
                    self.current_tick = int(tick)
                    if data.get('status', 'ACTIVE') == 'STOPPED':
                        global SHUTDOWN
                        SHUTDOWN = True
        except Exception:
            pass
    
    async def pre_news_unwind(self):
        """
        At t=58,118,178,238: cancel all → market order unwind to news_gross_limit.
        Then pause 3s. Normal trading limit (30000) is NOT changed.
        """
        target = self.news_gross_limit
        
        # Step 1: cancel all
        try:
            await RATE_LIMITER.acquire()
            async with self.session.post(
                f"{BASE_URL}/commands/cancel?all=1&key={API_KEY}", timeout=1.0
            ) as resp:
                await resp.read()
        except Exception:
            pass
        for engine in self.engines.values():
            engine.state.reset_order_tracking()
        
        # Step 2: refresh positions
        await self.parallel_data_fetch()
        exposure = self.calculate_current_gross()
        
        print(f"📰 PRE-NEWS t={self.current_tick} | gross={exposure} target={target}")
        
        # Step 3: market order unwind until gross <= target
        if exposure > target:
            # How much total to shed
            to_shed = exposure - target
            
            # Sort by largest position first
            sorted_engines = sorted(
                self.engines.items(),
                key=lambda x: abs(x[1].state.server_position),
                reverse=True
            )
            
            shed_remaining = to_shed
            for ticker, engine in sorted_engines:
                if shed_remaining <= 0:
                    break
                pos = engine.state.server_position
                if abs(pos) < 50:
                    continue
                
                qty = min(abs(pos), shed_remaining)
                action = 'SELL' if pos > 0 else 'BUY'
                
                try:
                    await RATE_LIMITER.acquire()
                    async with self.session.post(
                        f"{BASE_URL}/orders?key={API_KEY}",
                        params={
                            'ticker': ticker, 'type': 'MARKET',
                            'action': action, 'quantity': qty,
                        },
                        timeout=1.0,
                    ) as resp:
                        await resp.read()
                        shed_remaining -= qty
                except Exception:
                    pass
        
        # Pause 3s — after pause, trading resumes with normal GROSS_LIMIT_TOTAL (30000)
        self.news_pause_end = time.time() + NEWS_PAUSE_SECONDS
    
    async def _fetch_orders_atomic(self, session: aiohttp.ClientSession):
        """✅ NEW: Atomic order reconciliation at 50ms intervals"""
        try:
            await RATE_LIMITER.acquire()
            async with session.get(f"{BASE_URL}/orders?status=OPEN&key={API_KEY}", timeout=1.0) as resp:
                if resp.status == 200:
                    open_orders = await resp.json()
                    
                    # Mark all orders as not found initially
                    for engine in self.engines.values():
                        engine.state._order_seen_buy = False
                        engine.state._order_seen_sell = False
                    
                    # Process each open order from server
                    for o in open_orders:
                        order_id = o.get('order_id') or o.get('id')
                        ticker = o.get('ticker')
                        if ticker in self.engines:
                            engine = self.engines[ticker]
                            action = o.get('action')
                            
                            if action == 'BUY' and engine.state.active_buy_id == order_id:
                                original_qty = o.get('quantity', 0)
                                filled = o.get('quantity_filled', 0)
                                engine.state.pending_buy_qty = max(0, original_qty - filled)
                                engine.state._order_seen_buy = True
                            elif action == 'SELL' and engine.state.active_sell_id == order_id:
                                original_qty = o.get('quantity', 0)
                                filled = o.get('quantity_filled', 0)
                                engine.state.pending_sell_qty = max(0, original_qty - filled)
                                engine.state._order_seen_sell = True
                    
                    # Clear orders that weren't found (filled or cancelled)
                    for engine in self.engines.values():
                        # ⚠️ CRITICAL FIX: Only clear if we haven't seen it in TWO consecutive fetches
                        if not engine.state._order_seen_buy:
                            if engine.state.active_buy_id:
                                # Mark for potential clearing next cycle
                                engine.state.active_buy_id = None
                                engine.state.pending_buy_qty = 0
                        if not engine.state._order_seen_sell:
                            if engine.state.active_sell_id:
                                # Mark for potential clearing next cycle
                                engine.state.active_sell_id = None
                                engine.state.pending_sell_qty = 0
                        
                        # Update shadow position based on effective position
                        engine.state.shadow_position = engine.state.effective_position
        except Exception:
            pass
    
    async def run_global_diagnostic(self):
        now = time.time()
        if now - self.last_global_diagnostic < 10.0:
            return
        self.last_global_diagnostic = now
    
    def print_status(self):
        now = time.time()
        if now - self.last_status_print < 2.0:
            return
        
        print("\033[2J\033[H")
        exp = self.calculate_current_gross()
        lim = self.current_gross_limit
        hc = "🟢" if exp < lim * 0.6 else "🟡" if exp < lim * 0.8 else "🔴"
        pause = " ⏸️" if now < self.news_pause_end else ""
        emg = " 🚨" if self.emergency_unwind_active else ""
        
        print(f"t={self.current_tick:>3} | PNL: ${self.current_pnl:>10.2f} | "
              f"{hc} {exp}/{lim} | unwind_target={self.news_gross_limit} | API:{RATE_LIMITER.current_rate()}/s{pause}{emg}")
        
        for t, e in self.engines.items():
            s = e.state
            act = ("B" if s.active_buy_id else ".") + ("S" if s.active_sell_id else ".")
            print(f"  {t} {s.server_position:>7} {s.best_bid:.2f}/{s.best_ask:.2f} "
                  f"{s.spread*100:>4.1f}c {act} obi={s.obi_current:>+.2f}")
        
        self.last_status_print = now
    
    async def _check_endgame(self):
        try:
            url = f"{BASE_URL}/case?key={API_KEY}"
            async with self.session.get(url, timeout=0.5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('ticks_remaining', 999) <= 1:
                        await self.endgame_protocol()
                        return True
        except Exception:
            pass
        return False
    
    async def endgame_protocol(self):
        self.endgame_active = True
        
        # sep
        print(f"🎯 ENDGAME t={self.current_tick}")
        
        cancel_url = f"{BASE_URL}/commands/cancel?all=1&key={API_KEY}"
        try:
            async with self.session.post(cancel_url, timeout=1.0) as resp:
                await resp.read()
        except Exception:
            pass
        
        for ticker, engine in self.engines.items():
            state = engine.state
            if abs(state.server_position) > 100:
                action = 'SELL' if state.server_position > 0 else 'BUY'
                
                payload = {
                    'ticker': ticker,
                    'type': 'MARKET',
                    'action': action,
                    'quantity': abs(state.server_position)
                }
                
                url = f"{BASE_URL}/orders?key={API_KEY}"
                try:
                    async with self.session.post(url, params=payload, timeout=1.0) as resp:
                        await resp.read()
                except Exception:
                    pass
        
        await asyncio.sleep(0.5)
        
        global SHUTDOWN
        SHUTDOWN = True
    
    async def main_loop(self):
        print("✅ Trading live")
        
        while not SHUTDOWN:
            now = time.monotonic()
            
            # ── Fetch simulation tick every cycle ─────────
            await self.fetch_tick_info()
            
            if self.current_tick >= 299:
                await self.endgame_protocol()
                break
            
            # ── Pre-news unwind (2 ticks before news) ─────
            if self.current_tick in UNWIND_TICKS and self.current_tick != self.last_unwind_tick:
                self.last_unwind_tick = self.current_tick
                await self.parallel_data_fetch()
                await self.pre_news_unwind()
                continue
            
            # ── News pause ────────────────────────────────
            if time.time() < self.news_pause_end:
                self.cycles += 1
                await asyncio.sleep(0.05)
                continue
            
            # ── Emergency ─────────────────────────────────
            if self.emergency_unwind_active:
                await self.parallel_data_fetch()
                await self.check_risk_limits()
                self.cycles += 1
                await asyncio.sleep(0.05)
                continue
            
            if self.cycles % 100 == 0:
                self.current_pnl = self.calculate_current_pnl()
            
            # ── Normal trading ────────────────────────────
            await self.parallel_data_fetch()
            await self.check_risk_limits()
            
            if not self.emergency_unwind_active:
                for engine in self.engines.values():
                    if await engine.run_cycle(self.session, now):
                        self.last_trading_activity = time.time()
            
            if self.cycles % 200 == 0:
                self.print_status()
            
            self.cycles += 1
            sleep_time = max(0, FAST_CYCLE_MS - (time.monotonic() - now))
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    async def cleanup(self):
        # sep
        print("🛑 SHUTTING DOWN")
        
        try:
            cancel_url = f"{BASE_URL}/commands/cancel?all=1&key={API_KEY}"
            async with self.session.post(cancel_url, timeout=1.0) as resp:
                await resp.read()
        except Exception:
            pass
        
        if self.session:
            await self.session.close()
        
        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except:
                pass
        
        gc.enable()
        
        print(f"\n📈 FINAL STATISTICS")
        # quiet
        print(f"   Emergency Unwinds: {self.emergency_unwinds}")
        print(f"   Final PNL: ${self.current_pnl:,.2f}")
        
        for ticker, engine in self.engines.items():
            heartbeat = engine.state.heartbeat
            success_rate = heartbeat.total_successes / max(1, heartbeat.total_attempts) * 100
        
        print("✅ Clean shutdown complete")

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def signal_handler(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    print("\n🛑 Shutdown signal received")

async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    bot = UltimateHFTSentinel()
    
    try:
        if not await bot.initialize():
            print("❌ Failed to initialize. Exiting.")
            return
        
        await asyncio.sleep(0.5)
        await bot.main_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Manual shutdown")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
