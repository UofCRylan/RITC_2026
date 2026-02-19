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
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

API_KEY = 'ZVVCI8DU'
BASE_URL = 'http://localhost:9998/v1'
TICKERS = ['SPNG', 'SMMR', 'ATMN', 'WNTR']

OBI_THRESHOLD = 0.35
OBI_HISTORY_LENGTH = 15
OBI_BOOK_DEPTH = 40
OBI_MIN_ORDER_SIZE = 65000

TICKER_ALLOC = {'SPNG': 0.2, 'SMMR': 0.25, 'ATMN': 0.25, 'WNTR': 0.30}

CORRELATIONS = {
    ('SPNG', 'SMMR'):  0.6,  ('SMMR', 'SPNG'):  0.6,
    ('SMMR', 'ATMN'):  0.6,  ('ATMN', 'SMMR'):  0.6,
    ('ATMN', 'WNTR'):  0.6,  ('WNTR', 'ATMN'):  0.6,
}
CROSS_SIGNAL_WEIGHT = 0.25
HEDGE_STRENGTH = 0.6

GROSS_LIMIT = 0
REAL_GROSS_LIMIT = 0
TICKER_MAX: Dict[str, int] = {t: 0 for t in TICKERS}
TICKER_BASE: Dict[str, int] = {t: 0 for t in TICKERS}

MIN_SPREAD_FOR_AGGRESSIVE = 0.07
PRICE_UPDATE_THRESHOLD = 0.01
MIN_ORDER_SIZE = 50

TOKEN_BUCKET_CAPACITY = 50
TOKEN_REFILL_RATE_SEC = 0.001
FAST_CYCLE_MS = 0.03
BOOK_INTERVAL = 0.10
REQUEST_TIMEOUT = 0.15
API_RATE_LIMIT = 45

MAX_CONSECUTIVE_FAILURES = 5
STUCK_LOCK_TIMEOUT = 0.05
NO_TRADE_TIMEOUT = 30.0
MAX_RECOVERY_DURATION = 10.0
RECOVERY_COOLDOWN = 3.0
DRIFT_TOLERANCE = 500

GROSS_THROTTLE_START = 0.75
GROSS_THROTTLE_HARD  = 0.85
GROSS_CANCEL_THRESH  = 0.90
GROSS_EMERGENCY      = 0.95
GROSS_RESUME_FULL    = 0.80
EMERGENCY_COOLDOWN_S = 1.0
MAX_EMERGENCY_PER_30 = 3

DAY_LENGTH = 60
DAY_BOUNDARIES = [60, 120, 180, 240, 300]
FLATTEN_LEAD = 0.5
FLATTEN_RAMP_LEAD = 8
FLATTEN_MARKET_SAFETY = 500
PAUSE_AFTER_BOUNDARY = 0.5

SHUTDOWN = False

URL_SEC    = f"{BASE_URL}/securities?key={API_KEY}"
URL_ORD_O  = f"{BASE_URL}/orders?status=OPEN&key={API_KEY}"
URL_ORD    = f"{BASE_URL}/orders?key={API_KEY}"
URL_CXL    = f"{BASE_URL}/commands/cancel?all=1&key={API_KEY}"
URL_CASE   = f"{BASE_URL}/case?key={API_KEY}"
URL_NEWS   = f"{BASE_URL}/news?key={API_KEY}"
def url_book(t): return f"{BASE_URL}/securities/book?ticker={t}&key={API_KEY}&limit={OBI_BOOK_DEPTH}"
def url_cxl_t(t): return f"{BASE_URL}/commands/cancel?ticker={t}&key={API_KEY}"
def url_del(i): return f"{BASE_URL}/orders/{i}?key={API_KEY}"

class RL_:
    __slots__ = ('max_ps', 'times', 'blocked')
    def __init__(self, mps=API_RATE_LIMIT):
        self.max_ps = mps; self.times: deque = deque(); self.blocked = 0
    async def acquire(self):
        now = time.monotonic(); t = self.times
        while t and t[0] < now - 1.0: t.popleft()
        if len(t) >= self.max_ps:
            w = t[0] + 1.0 - now + 0.002
            if w > 0: self.blocked += 1; await asyncio.sleep(w)
        t.append(time.monotonic())
    def rate(self) -> int:
        now = time.monotonic(); t = self.times
        while t and t[0] < now - 1.0: t.popleft()
        return len(t)

RL = RL_()

@dataclass
class Heartbeat:
    last_attempt: float = field(default_factory=time.time)
    last_success: float = field(default_factory=time.time)
    consec_fail: int = 0; total_att: int = 0; total_ok: int = 0; recovery_act: int = 0
    def record(self, ok):
        self.last_attempt = time.time(); self.total_att += 1
        if ok: self.last_success = time.time(); self.consec_fail = 0; self.total_ok += 1
        else: self.consec_fail += 1
    def is_stuck(self, in_rec=False):
        if in_rec: return self.consec_fail >= 8 or time.time() - self.last_success > 15.0
        if self.consec_fail >= MAX_CONSECUTIVE_FAILURES: return True
        return time.time() - self.last_success > NO_TRADE_TIMEOUT and time.time() - self.last_attempt < 10.0

@dataclass
class Bucket:
    tokens: int = TOKEN_BUCKET_CAPACITY
    last_refill: float = field(default_factory=time.monotonic)
    last_checked: float = field(default_factory=time.monotonic)
    empty_streak: int = 0
    def consume(self):
        now = time.monotonic()
        if now - self.last_checked > 0.1:
            r = int((now - self.last_refill) / TOKEN_REFILL_RATE_SEC)
            if r > 0: self.tokens = min(TOKEN_BUCKET_CAPACITY, self.tokens + r); self.last_refill = now
            self.last_checked = now
        if self.tokens >= 1: self.tokens -= 1; self.empty_streak = 0; return True
        self.empty_streak += 1; return False
    def force(self): self.tokens = TOKEN_BUCKET_CAPACITY; self.last_refill = time.monotonic(); self.empty_streak = 0
    def is_stuck(self): return self.empty_streak > 20 and self.tokens == 0

@dataclass
class TState:
    ticker: str
    server_position: int = 0
    best_bid: float = 0.0; best_ask: float = 0.0; spread: float = 0.0
    realized_pnl: float = 0.0; last_price: float = 0.0
    shadow_position: int = 0
    pending_buy_qty: int = 0; pending_sell_qty: int = 0
    active_buy_id: Optional[int] = None; active_sell_id: Optional[int] = None
    active_buy_price: float = 0.0; active_sell_price: float = 0.0
    obi_current: float = 0.0
    obi_history: deque = field(default_factory=lambda: deque(maxlen=OBI_HISTORY_LENGTH))
    obi_delta: float = 0.0; market_type: str = "FLAT"
    cross_signal: float = 0.0
    hedge_bias: float = 0.0
    buy_in_flight: bool = False; sell_in_flight: bool = False
    buy_lock_time: float = 0.0; sell_lock_time: float = 0.0
    bucket: Bucket = field(default_factory=Bucket)
    cycles_since_drift: int = 0; order_sizes_reduced: bool = False
    hb: Heartbeat = field(default_factory=Heartbeat)
    in_recovery: bool = False; recovery_start: float = 0.0; recovery_cooldown_end: float = 0.0
    _sb: bool = False; _ss: bool = False
    # ── PESSIMISTIC SHADOW TRACKING ──
    # placed_*_size: size of current OPEN order (0 or size, max 1 per side)
    # pending_fill_*: accumulated fills not yet confirmed by server
    # shadow = server + (placed_buy + pending_fill_buy) - (placed_sell + pending_fill_sell)
    buy_fill_pending: bool = False   # UNUSED — kept for compat
    sell_fill_pending: bool = False
    fill_pending_time: float = 0.0
    placed_buy_size: int = 0
    placed_sell_size: int = 0
    pending_fill_buy: int = 0    # fills waiting for server confirmation
    pending_fill_sell: int = 0

    @property
    def shadow_position(self):
        return self.server_position + self.placed_buy_size + self.pending_fill_buy - self.placed_sell_size - self.pending_fill_sell
    @shadow_position.setter 
    def shadow_position(self, val):
        pass  # no-op — shadow is computed

    @property
    def wc_long(self): return self.server_position + self.pending_buy_qty
    @property
    def wc_short(self): return self.server_position - self.pending_sell_qty
    @property
    def effective_position(self):
        wl, ws = self.wc_long, self.wc_short
        return wl if abs(wl) >= abs(ws) else ws
    def worst_case_gross(self):
        pos = self.server_position
        return max(abs(pos + self.pending_buy_qty), abs(pos - self.pending_sell_qty))
    def conservative_exposure(self):
        return abs(self.server_position) + self.pending_buy_qty + self.pending_sell_qty
    def is_reducing(self, action):
        # Use shadow_position (our best local estimate) not server_position (stale)
        pos = self.shadow_position
        if action == 'BUY' and pos < 0: return True
        if action == 'SELL' and pos > 0: return True
        return False
    def reset_orders(self):
        self.pending_buy_qty = 0; self.pending_sell_qty = 0
        self.active_buy_id = None; self.active_sell_id = None
        self.active_buy_price = 0.0; self.active_sell_price = 0.0
        self.buy_in_flight = False; self.sell_in_flight = False
        self.buy_lock_time = 0.0; self.sell_lock_time = 0.0
        self.buy_fill_pending = False; self.sell_fill_pending = False; self.fill_pending_time = 0.0
        self.placed_buy_size = 0; self.placed_sell_size = 0
        self.pending_fill_buy = 0; self.pending_fill_sell = 0
        self.shadow_position = self.server_position; self.bucket.force()
    def check_stuck(self):
        now = time.time()
        if self.buy_in_flight and now - self.buy_lock_time > STUCK_LOCK_TIMEOUT: return True
        if self.sell_in_flight and now - self.sell_lock_time > STUCK_LOCK_TIMEOUT: return True
        if self.bucket.is_stuck(): return True
        return now > self.recovery_cooldown_end and self.hb.is_stuck(self.in_recovery)

def calc_obi(book):
    try:
        bids = book.get('bids', book.get('bid', []))
        asks = book.get('asks', book.get('ask', []))
        bv = sum(int(b.get('quantity', 0)) for b in bids[:OBI_BOOK_DEPTH] if int(b.get('quantity', 0)) >= OBI_MIN_ORDER_SIZE)
        av = sum(int(a.get('quantity', 0)) for a in asks[:OBI_BOOK_DEPTH] if int(a.get('quantity', 0)) >= OBI_MIN_ORDER_SIZE)
        tot = bv + av
        return max(-1.0, min(1.0, (bv - av) / tot)) if tot else 0.0
    except: return 0.0

def obi_delta(hist):
    if len(hist) < 10: return 0.0
    h = list(hist)
    return sum(h[-5:]) / 5.0 - sum(h[-10:-5]) / 5.0

def mtype(obi):
    if abs(obi) <= OBI_THRESHOLD: return "FLAT"
    return "BULL" if obi > 0 else "BEAR"

def calc_prices(st, sv_gross_ratio, reducing_only=False, boundary_scale=1.0):
    tk = st.ticker; base = TICKER_BASE.get(tk, 0); limit = TICKER_MAX.get(tk, 1)
    if limit <= 0: limit = 1
    pos = st.effective_position
    ratio = max(-1.0, min(1.0, pos / limit))
    bsk = round(1.0 - ratio, 2); ssk = 2.0 - bsk
    bsz = int(base * bsk); ssz = int(base * ssk)
    if bsz >= 100: bsz = round(bsz, -1)
    if ssz >= 100: ssz = round(ssz, -1)
    if st.in_recovery: bsz = int(bsz * 0.2); ssz = int(ssz * 0.2)
    elif st.order_sizes_reduced and st.cycles_since_drift < 3: bsz = int(bsz * 0.5); ssz = int(ssz * 0.5)
    hb = st.hedge_bias
    skip_hedge = reducing_only or (abs(pos) > limit * 0.85)
    if not skip_hedge and abs(hb) > 0.1:
        strength = min(1.0, abs(hb)) * HEDGE_STRENGTH
        if hb > 0: bsz = int(bsz * (1.0 + strength * 0.5)); ssz = int(ssz * (1.0 - strength * 0.8))
        else: bsz = int(bsz * (1.0 - strength * 0.8)); ssz = int(ssz * (1.0 + strength * 0.5))
    # ── SOFT POSITION CAP: scale down building-side as we approach limit ──
    apos = abs(pos)
    if apos > limit * 0.80:
        cap_scale = max(0.15, (limit - apos) / (limit * 0.20))
        if pos > 0: bsz = int(bsz * cap_scale)    # shrink buys when long
        elif pos < 0: ssz = int(ssz * cap_scale)   # shrink sells when short
    # ── ANTI-ACCUMULATION BRAKE: if OBI aligns with position, shrink building-side early ──
    if apos > limit * 0.40:
        if pos > 0 and st.market_type == "BULL":
            brake = max(0.3, 1.0 - (apos / limit))   # at 40%→0.6x, at 80%→0.2x
            bsz = int(bsz * brake)
        elif pos < 0 and st.market_type == "BEAR":
            brake = max(0.3, 1.0 - (apos / limit))
            ssz = int(ssz * brake)
    if boundary_scale < 1.0:
        if pos > 0:
            bsz = int(bsz * boundary_scale)
            ssz = int(ssz * max(1.0, 2.0 - boundary_scale))
        elif pos < 0:
            ssz = int(ssz * boundary_scale)
            bsz = int(bsz * max(1.0, 2.0 - boundary_scale))
        else:
            bsz = int(bsz * boundary_scale)
            ssz = int(ssz * boundary_scale)
    if sv_gross_ratio > GROSS_THROTTLE_START:
        if sv_gross_ratio >= GROSS_THROTTLE_HARD:
            if pos > 0: bsz = 0
            elif pos < 0: ssz = 0
            else: bsz = 0; ssz = 0
        else:
            throttle = max(0.1, 1.0 - ((sv_gross_ratio - GROSS_THROTTLE_START) / (GROSS_THROTTLE_HARD - GROSS_THROTTLE_START)))
            bsz = int(bsz * throttle); ssz = int(ssz * throttle)
    if reducing_only:
        if pos > 0: bsz = 0
        elif pos < 0: ssz = 0
        else: bsz = 0; ssz = 0
    if pos > 0: bsz = min(bsz, max(0, limit - pos))
    elif pos < 0: ssz = min(ssz, max(0, limit - abs(pos)))
    bsz = max(MIN_ORDER_SIZE, bsz) if bsk >= 0.05 and bsz >= MIN_ORDER_SIZE else 0
    ssz = max(MIN_ORDER_SIZE, ssz) if ssk >= 0.05 and ssz >= MIN_ORDER_SIZE else 0
    # ── POSITION-AWARE PRICING: aggression used for REDUCING only ──
    # Old: BULL → always aggressive buy. This BUILDS positions in trends.
    # New: BULL + long → passive buy / aggressive sell (REDUCE the long)
    #      BULL + short → aggressive buy (REDUCE the short) / passive sell
    #      BEAR mirrors this logic.
    if st.market_type == "BULL":
        if pos > 0:
            # Already long in a BULL market — DON'T chase, help reduce
            bp = st.best_bid                         # passive buy (earn rebate)
            sp = round(st.best_ask - 0.01, 2)        # aggressive sell (reduce long)
        elif pos < 0:
            # Short in BULL — aggressively buy to reduce short
            bp = round(st.best_bid + 0.01, 2)        # aggressive buy (reduce short)
            sp = round(st.best_ask + 0.01, 2)         # passive sell (earn rebate)
        else:
            bp = round(st.best_bid + 0.01, 2); sp = round(st.best_ask + 0.01, 2)
    elif st.market_type == "BEAR":
        if pos < 0:
            # Already short in a BEAR market — DON'T chase, help reduce
            sp = st.best_ask                          # passive sell (earn rebate)
            bp = round(st.best_bid + 0.01, 2)         # aggressive buy (reduce short)
        elif pos > 0:
            # Long in BEAR — aggressively sell to reduce long
            sp = round(st.best_ask - 0.01, 2)         # aggressive sell (reduce long)
            bp = round(st.best_bid - 0.01, 2)         # passive buy (earn rebate)
        else:
            sp = round(st.best_ask - 0.01, 2); bp = round(st.best_bid - 0.01, 2)
    else:
        if st.spread >= MIN_SPREAD_FOR_AGGRESSIVE: bp = round(st.best_bid + 0.01, 2); sp = round(st.best_ask - 0.01, 2)
        else: bp = st.best_bid; sp = st.best_ask
    cs = st.cross_signal
    if abs(cs) > 0.15:
        adj = round(cs * CROSS_SIGNAL_WEIGHT * max(st.spread, 0.02), 2)
        bp = round(bp + adj, 2); sp = round(sp + adj, 2)
    if bp >= sp: bp = round(sp - 0.01, 2)
    return bp, sp, int(bsz), int(ssz)

class Engine:
    __slots__ = ('ticker', 'parent', 'state', 'last_book_fetch', 'cycle_count', 'rec_attempts', 'last_trade_time', 'consec_orders')
    def __init__(self, ticker, parent):
        self.ticker = ticker; self.parent = parent; self.state = TState(ticker=ticker)
        self.last_book_fetch = 0.0; self.cycle_count = 0; self.rec_attempts = 0
        self.last_trade_time = 0.0; self.consec_orders = 0

    def can_place(self, action, qty):
        now = time.time()
        if now < self.state.recovery_cooldown_end or now - self.last_trade_time < 0.05: return False
        s = self.state
        if self.parent.emergency_active:
            if not s.is_reducing(action): return False
        if action == 'BUY':
            if s.buy_in_flight:
                if now - s.buy_lock_time > 1.0: s.buy_in_flight = False
                else: return False
        else:
            if s.sell_in_flight:
                if now - s.sell_lock_time > 1.0: s.sell_in_flight = False
                else: return False
        if not s.bucket.consume():
            if s.bucket.is_stuck(): s.bucket.force()
            return False
        if REAL_GROSS_LIMIT <= 0: return True

        # ── ABSOLUTE GROSS CAP (never bypassed) ──
        total_gross = self.parent.gross_shadow()
        if total_gross > REAL_GROSS_LIMIT:
            if not s.is_reducing(action): return False
        if total_gross > REAL_GROSS_LIMIT * 1.05:
            return False  # hard stop even for reducing

        # ── PER-TICKER CHECK (uses shadow for best local estimate) ──
        tk_limit = TICKER_MAX.get(self.ticker, 1)
        pos = s.shadow_position
        if not s.is_reducing(action):
            if action == 'BUY': projected = pos + qty
            else: projected = pos - qty
            if abs(projected) > tk_limit: self.parent.gate_blocks += 1; return False

        # ── GROSS WC CHECK (building orders only — reducing orders always allowed) ──
        # Reducing orders decrease gross position. Blocking them creates a death spiral
        # where the bot can't unwind and stays stuck at 100%+ for 30+ ticks.
        if not s.is_reducing(action):
            wc_gross = 0
            for tk, eng in self.parent.engines.items():
                es = eng.state
                if tk == self.ticker:
                    if action == 'BUY': wc_gross += max(abs(pos + qty), abs(pos))
                    else: wc_gross += max(abs(pos), abs(pos - qty))
                else: wc_gross += abs(es.shadow_position)
            if wc_gross > REAL_GROSS_LIMIT * 0.88: self.parent.gate_blocks += 1; return False
        return True

    async def _execute(self, session, side, price, size):
        s = self.state
        if side == 'BUY': s.buy_in_flight = True; s.buy_lock_time = time.time(); old = s.active_buy_id
        else: s.sell_in_flight = True; s.sell_lock_time = time.time(); old = s.active_sell_id
        ok = False
        try:
            if old is not None:
                try:
                    await RL.acquire()
                    async with session.delete(url_del(old), timeout=0.10) as r: await r.read()
                except: pass
            await RL.acquire()
            async with session.post(URL_ORD, params={'ticker': self.ticker, 'type': 'LIMIT', 'action': side, 'quantity': size, 'price': price}, timeout=0.5) as r:
                if r.status == 200:
                    d = await r.json(); oid = d.get('order_id') or d.get('id')
                    t = d.get('tick')
                    if t is not None: self.parent.current_tick = max(self.parent.current_tick, int(t))
                    filled = int(d.get('quantity_filled', 0))

                    # ── PESSIMISTIC SHADOW: overwrite, never accumulate ──
                    # Max 1 buy + 1 sell order per ticker. placed = that ONE order.
                    if side == 'BUY':
                        s.active_buy_id = oid; s.active_buy_price = price
                        s.pending_buy_qty = max(0, size - filled)
                        s.placed_buy_size = size  # always overwrite
                    else:
                        s.active_sell_id = oid; s.active_sell_price = price
                        s.pending_sell_qty = max(0, size - filled)
                        s.placed_sell_size = size  # always overwrite
                    s.hb.record(True); ok = True; self.rec_attempts = 0
                    self.last_trade_time = time.time(); self.consec_orders += 1
                    if self.consec_orders > 10: self.consec_orders = 0; await asyncio.sleep(0.01)
                else: raise Exception()
        except: s.bucket.force(); s.hb.record(False); self.consec_orders = 0
        finally:
            if side == 'BUY': s.buy_in_flight = False
            else: s.sell_in_flight = False
        return ok

    async def enter_recovery(self, session):
        if self.state.in_recovery: return
        self.state.in_recovery = True; self.state.recovery_start = time.time()
        self.state.recovery_cooldown_end = time.time() + 1.0; self.rec_attempts += 1
        try:
            await RL.acquire()
            async with session.post(url_cxl_t(self.ticker), timeout=1.0) as r: await r.read()
        except: pass
        self.state.reset_orders(); self.state.hb.recovery_act += 1; await asyncio.sleep(0.1)
    def check_exit_recovery(self):
        if not self.state.in_recovery: return False
        now = time.time(); dur = now - self.state.recovery_start
        if dur > MAX_RECOVERY_DURATION or (self.state.hb.consec_fail == 0 and now - self.state.hb.last_success < 5.0) or (dur > 3.0 and self.state.hb.consec_fail < 3):
            self.state.in_recovery = False; self.state.recovery_cooldown_end = now + RECOVERY_COOLDOWN; return True
        return False
    def has_drift(self):
        if self.state.pending_buy_qty == 0 and self.state.pending_sell_qty == 0: return False
        return abs(self.state.shadow_position - self.state.server_position) > DRIFT_TOLERANCE and self.state.cycles_since_drift > 30
    async def fix_drift(self, session):
        if self.state.active_buy_id or self.state.active_sell_id:
            try:
                await RL.acquire()
                async with session.post(url_cxl_t(self.ticker), timeout=0.5) as r: await r.read()
            except: pass
        self.state.reset_orders(); self.state.order_sizes_reduced = True; self.state.cycles_since_drift = 0; await asyncio.sleep(0.01)
    async def run_cycle(self, session, sv_gross_ratio, reducing_only=False, boundary_scale=1.0):
        if self.parent.endgame_active: return False
        if self.state.spread <= 0: return False
        self.cycle_count += 1; self.state.cycles_since_drift += 1
        if time.time() - self.last_trade_time > 1.0: self.consec_orders = 0
        if self.state.in_recovery:
            if self.check_exit_recovery(): self.state.reset_orders(); return False
        if self.state.check_stuck() and not self.state.in_recovery and self.cycle_count > 50 and time.time() > self.state.recovery_cooldown_end:
            if self.rec_attempts < 3: await self.enter_recovery(session)
            else: self.state.recovery_cooldown_end = time.time() + 30.0
            return False
        if self.has_drift(): await self.fix_drift(session); return True
        if self.state.order_sizes_reduced and self.state.cycles_since_drift >= 3: self.state.order_sizes_reduced = False
        bp, sp, bsz, ssz = calc_prices(self.state, sv_gross_ratio, reducing_only, boundary_scale)
        placed = 0
        if bsz >= MIN_ORDER_SIZE and (self.state.active_buy_price == 0.0 or abs(bp - self.state.active_buy_price) >= PRICE_UPDATE_THRESHOLD):
            if self.can_place('BUY', bsz):
                if await self._execute(session, 'BUY', bp, bsz): placed += 1
        if ssz >= MIN_ORDER_SIZE and (self.state.active_sell_price == 0.0 or abs(sp - self.state.active_sell_price) >= PRICE_UPDATE_THRESHOLD):
            if self.can_place('SELL', ssz):
                if await self._execute(session, 'SELL', sp, ssz): placed += 1
        return placed > 0

class SeasonalMM:
    def __init__(self):
        if sys.platform == 'win32':
            try: import ctypes; ctypes.windll.winmm.timeBeginPeriod(1)
            except: pass
            if sys.version_info >= (3, 8): asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        gc.disable()
        try: psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)
        except: pass
        self.engines: Dict[str, Engine] = {t: Engine(t, self) for t in TICKERS}
        self.session = None; self.cycles = 0; self.current_pnl = 0.0
        self.emergency_active = False; self.endgame_active = False
        self.last_status = 0.0; self.last_trading_activity = time.time()
        self.current_tick = 0; self._book_idx = 0; self._last_book = 0.0; self._last_ord = 0.0
        self._news_found = False
        self.emergency_count = 0; self.last_emergency_time = 0.0
        self.emergency_times: deque = deque(maxlen=10)
        self.gate_blocks = 0; self.cancel_events = 0; self.full_cancel_events = 0
        self.flatten_active = False; self.day_pause_active = False; self.last_flatten_boundary = -1

    def gross_server(self): return sum(abs(e.state.server_position) for e in self.engines.values())
    def gross_shadow(self): return sum(abs(e.state.shadow_position) for e in self.engines.values())
    def gross_worst_case(self): return sum(e.state.worst_case_gross() for e in self.engines.values())
    def wc_gross_ratio(self):
        if REAL_GROSS_LIMIT <= 0: return 0.0
        return self.gross_worst_case() / REAL_GROSS_LIMIT
    def server_gross_ratio(self):
        if REAL_GROSS_LIMIT <= 0: return 0.0
        return self.gross_server() / REAL_GROSS_LIMIT
    def pnl(self): return sum(e.state.realized_pnl for e in self.engines.values())

    def next_boundary(self):
        for b in DAY_BOUNDARIES:
            if self.current_tick < b: return b
        return DAY_BOUNDARIES[-1]
    def ticks_to_boundary(self): return self.next_boundary() - self.current_tick
    def day_phase(self):
        for b in DAY_BOUNDARIES[:-1]:
            if b <= self.current_tick <= b + PAUSE_AFTER_BOUNDARY:
                return 'pause'
        ttb = self.ticks_to_boundary()
        if 0 < ttb <= FLATTEN_LEAD: return 'flatten'
        if 0 < ttb <= FLATTEN_RAMP_LEAD: return 'ramp_down'
        return 'normal'

    async def flatten_for_day_end(self):
        nb = self.next_boundary(); ttb = self.ticks_to_boundary()
        if self.last_flatten_boundary != nb:
            self.last_flatten_boundary = nb; self.flatten_active = True
            total_pos = sum(abs(e.state.server_position) for e in self.engines.values())
            print(f"🌙 FLATTEN: t={self.current_tick} ttb={ttb} (gross={total_pos})")
            try:
                await RL.acquire()
                async with self.session.post(URL_CXL, timeout=1.0) as r: await r.read()
            except: pass
            for e in self.engines.values(): e.state.reset_orders()
            await asyncio.sleep(0.05); await self._fetch_securities()
            for tk, e in self.engines.items():
                pos = e.state.server_position
                if abs(pos) <= 10: continue
                if pos > 0: price = round(e.state.best_ask, 2); action = 'SELL'
                else: price = round(e.state.best_bid, 2); action = 'BUY'
                try:
                    await RL.acquire()
                    async with self.session.post(URL_ORD, params={
                        'ticker': tk, 'type': 'LIMIT', 'action': action,
                        'quantity': abs(pos), 'price': price
                    }, timeout=0.5) as r:
                        if r.status == 200: print(f"  📋 {tk}: {action} {abs(pos)} @ {price:.2f} (passive)")
                except: pass
        if ttb <= 1 and self.last_flatten_boundary == nb:  # only fire once per boundary
            if not hasattr(self, '_safety_boundary') or self._safety_boundary != nb:
                await self._fetch_securities()
                total_pos = sum(abs(e.state.server_position) for e in self.engines.values())
                if total_pos > FLATTEN_MARKET_SAFETY:
                    self._safety_boundary = nb
                    print(f"  🔥 MARKET SAFETY: ttb={ttb} gross={total_pos}")
                    try:
                        await RL.acquire()
                        async with self.session.post(URL_CXL, timeout=0.5) as r: await r.read()
                    except: pass
                    for tk, e in self.engines.items():
                        pos = e.state.server_position
                        if abs(pos) > 50:
                            act = 'SELL' if pos > 0 else 'BUY'
                            try:
                                await RL.acquire()
                                async with self.session.post(URL_ORD, params={
                                    'ticker': tk, 'type': 'MARKET', 'action': act, 'quantity': abs(pos)
                                }, timeout=1.0) as r: await r.read()
                            except: pass
                    for e in self.engines.values(): e.state.reset_orders()

    async def day_pause(self):
        if not self.day_pause_active:
            self.day_pause_active = True
            try:
                await RL.acquire()
                async with self.session.post(URL_CXL, timeout=1.0) as r: await r.read()
            except: pass
            for e in self.engines.values(): e.state.reset_orders()
            total_pos = sum(abs(e.state.server_position) for e in self.engines.values())
            print(f"  ⏸️  PAUSED at t={self.current_tick} (pos={total_pos})")
    def resume_from_pause(self):
        if self.day_pause_active:
            self.day_pause_active = False; self.flatten_active = False
            for tk, e in self.engines.items():
                tk_limit = TICKER_MAX.get(tk, 1)
                if abs(e.state.server_position) > tk_limit:
                    e.state.order_sizes_reduced = True; e.state.cycles_since_drift = 0
                    print(f"  ⚠️  {tk} overweight: {e.state.server_position}/{tk_limit} → reduced mode")
            print(f"  ▶️  RESUMED at t={self.current_tick}")

    async def initialize(self):
        conn = aiohttp.TCPConnector(limit=50, limit_per_host=50, family=socket.AF_INET, ttl_dns_cache=3600, force_close=False, enable_cleanup_closed=True)
        self.session = aiohttp.ClientSession(connector=conn, json_serialize=ujson.dumps, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT))
        for att in range(3):
            try:
                async with self.session.get(URL_CASE, timeout=1.0) as r:
                    if r.status == 200:
                        d = await r.json(); t = d.get('tick')
                        if t is None: t = 300 - d.get('ticks_remaining', 300)
                        self.current_tick = int(t)
                        print(f"✅ API connected | tick={self.current_tick}"); break
                    elif att == 2: print("❌ API failed"); return False
            except:
                if att < 2: await asyncio.sleep(0.5)
                else: return False
        await self._fetch_securities()
        for e in self.engines.values(): e.state.shadow_position = e.state.server_position
        await self._scrape_news(); return True

    async def _scrape_news(self):
        global GROSS_LIMIT, REAL_GROSS_LIMIT, TICKER_MAX, TICKER_BASE
        for att in range(10):
            try:
                await RL.acquire()
                async with self.session.get(URL_NEWS, timeout=aiohttp.ClientTimeout(total=2.0)) as r:
                    if r.status != 200: await asyncio.sleep(0.5); continue
                    data = await r.json()
                    if not data or len(data) < 2: print(f"📰 Attempt {att+1}: {len(data) if data else 0} items..."); await asyncio.sleep(1.0); continue
                    news = sorted(data, key=lambda x: x.get('news_id', x.get('id', 0)))
                    for i, it in enumerate(news):
                        txt = f"{it.get('headline', '')} {it.get('body', '')}"; print(f"📰 [{i}]: {txt[:150]}")
                    for item in [news[1]] + news:
                        txt = f"{item.get('headline', '')} {item.get('body', '')}"
                        m = re.search(r'([\d,]+)\s+shares', txt, re.IGNORECASE)
                        if m:
                            raw = int(m.group(1).replace(',', '')); REAL_GROSS_LIMIT = raw
                            GROSS_LIMIT = raw - 500  # minimal buffer — gate handles the rest
                            for tk in TICKERS:
                                TICKER_MAX[tk] = int(GROSS_LIMIT * TICKER_ALLOC[tk])
                                TICKER_BASE[tk] = max(50, TICKER_MAX[tk] // 3)  # aggressive sizing
                            self._news_found = True
                            alloc_str = " | ".join(f"{tk}={TICKER_MAX[tk]:,}(b={TICKER_BASE[tk]})" for tk in TICKERS)
                            print(f"✅ NEWS={raw:,} REAL={REAL_GROSS_LIMIT:,} | {alloc_str}"); return
                    print(f"⚠️  Attempt {att+1}: 'shares' not found"); await asyncio.sleep(1.0)
            except Exception as e: print(f"⚠️  News error: {e}"); await asyncio.sleep(0.5)
        REAL_GROSS_LIMIT = 9000; GROSS_LIMIT = 9000 - 500
        for tk in TICKERS: TICKER_MAX[tk] = int(GROSS_LIMIT * TICKER_ALLOC[tk]); TICKER_BASE[tk] = max(50, TICKER_MAX[tk] // 3)
        print(f"⚠️  Fallback: REAL={REAL_GROSS_LIMIT} GROSS={GROSS_LIMIT}")

    async def _fetch_case(self):
        try:
            await RL.acquire()
            async with self.session.get(URL_CASE, timeout=0.5) as r:
                if r.status == 200:
                    d = await r.json(); t = d.get('tick')
                    if t is None: t = d.get('current_tick')
                    if t is None: t = 300 - d.get('ticks_remaining', 300)
                    if t is not None: self.current_tick = max(self.current_tick, int(t))
        except: pass

    async def _fetch_securities(self):
        try:
            await RL.acquire()
            async with self.session.get(URL_SEC, timeout=0.5) as r:
                if r.status == 200:
                    for d in await r.json():
                        tk = d.get('ticker')
                        if tk in self.engines:
                            s = self.engines[tk].state
                            # ── CLEAR FILL PENDING: only when server ACTUALLY moved ──
                            old_pos = s.server_position
                            new_pos = int(d.get('position', 0))
                            s.server_position = new_pos
                            s.best_bid = float(d.get('bid', 0.0)); s.best_ask = float(d.get('ask', 0.0))
                            s.spread = s.best_ask - s.best_bid; s.realized_pnl = float(d.get('realized', 0.0))
                            s.last_price = float(d.get('last', 0.0))
                            # ── When server position changes, it has absorbed our orders ──
                            if new_pos != old_pos:
                                s.placed_buy_size = 0; s.placed_sell_size = 0
                                s.pending_fill_buy = 0; s.pending_fill_sell = 0
        except: pass
        # ── SAFETY NET: cancel building-side if gross approaching real limit ──
        if REAL_GROSS_LIMIT > 0:
            total_gross = self.gross_server()
            if total_gross > REAL_GROSS_LIMIT * 0.95:
                for tk, e in self.engines.items():
                    pos = e.state.server_position
                    if pos > 0 and e.state.active_buy_id:
                        try:
                            await RL.acquire()
                            async with self.session.delete(url_del(e.state.active_buy_id), timeout=0.10) as r: await r.read()
                            e.state.active_buy_id = None; e.state.pending_buy_qty = 0
                        except: pass
                    elif pos < 0 and e.state.active_sell_id:
                        try:
                            await RL.acquire()
                            async with self.session.delete(url_del(e.state.active_sell_id), timeout=0.10) as r: await r.read()
                            e.state.active_sell_id = None; e.state.pending_sell_qty = 0
                        except: pass

    async def _fetch_book(self):
        now = time.monotonic()
        if now - self._last_book < BOOK_INTERVAL: return
        self._last_book = now; tk = TICKERS[self._book_idx % len(TICKERS)]; self._book_idx += 1
        try:
            await RL.acquire()
            async with self.session.get(url_book(tk), timeout=0.5) as r:
                if r.status == 200:
                    bd = await r.json(); obi = calc_obi(bd); s = self.engines[tk].state
                    s.obi_history.append(obi); s.obi_current = obi
                    s.obi_delta = obi_delta(s.obi_history); s.market_type = mtype(obi)
                    self.engines[tk].last_book_fetch = now
        except: pass
    async def _fetch_orders(self):
        now = time.monotonic()
        if now - self._last_ord < 0.05: return
        self._last_ord = now
        try:
            await RL.acquire()
            async with self.session.get(URL_ORD_O, timeout=1.0) as r:
                if r.status == 200:
                    orders = await r.json()
                    for e in self.engines.values(): e.state._sb = False; e.state._ss = False
                    for o in orders:
                        t = o.get('tick')
                        if t is not None: self.current_tick = max(self.current_tick, int(t))
                        oid = o.get('order_id') or o.get('id'); tk = o.get('ticker')
                        if tk in self.engines:
                            e = self.engines[tk]; act = o.get('action')
                            rem = max(0, o.get('quantity', 0) - o.get('quantity_filled', 0))
                            if act == 'BUY' and e.state.active_buy_id == oid: e.state.pending_buy_qty = rem; e.state._sb = True
                            elif act == 'SELL' and e.state.active_sell_id == oid: e.state.pending_sell_qty = rem; e.state._ss = True
                    for e in self.engines.values():
                        if not e.state._sb and e.state.active_buy_id:
                            # Buy order disappeared — likely filled, track as pending
                            e.state.pending_fill_buy += e.state.placed_buy_size
                            e.state.active_buy_id = None; e.state.pending_buy_qty = 0
                            e.state.placed_buy_size = 0
                        elif e.state._sb:
                            e.state.placed_buy_size = e.state.pending_buy_qty  # sync to remaining
                        if not e.state._ss and e.state.active_sell_id:
                            # Sell order disappeared — likely filled
                            e.state.pending_fill_sell += e.state.placed_sell_size
                            e.state.active_sell_id = None; e.state.pending_sell_qty = 0
                            e.state.placed_sell_size = 0
                        elif e.state._ss:
                            e.state.placed_sell_size = e.state.pending_sell_qty  # sync to remaining
        except: pass

    async def fetch_all(self):
        await asyncio.gather(self._fetch_case(), self._fetch_securities(), self._fetch_book(), self._fetch_orders(), return_exceptions=True)
        self._update_cross_signals(); self._update_hedge_bias()

    def _update_cross_signals(self):
        for tk in TICKERS:
            signals = []
            for (a, b), corr in CORRELATIONS.items():
                if a == tk and b in self.engines:
                    other_obi = self.engines[b].state.obi_current
                    if other_obi != 0.0: signals.append(corr * other_obi)
            self.engines[tk].state.cross_signal = sum(signals) / len(signals) if signals else 0.0
    def _update_hedge_bias(self):
        for tk in TICKERS:
            bias = 0.0
            for (a, b), rho in CORRELATIONS.items():
                if a == tk and b in self.engines:
                    neighbor_pos = self.engines[b].state.server_position
                    neighbor_limit = TICKER_MAX.get(b, 1) or 1
                    if neighbor_pos == 0: continue
                    sign_p = 1 if neighbor_pos > 0 else -1
                    magnitude = min(1.0, abs(neighbor_pos) / neighbor_limit)
                    bias += -rho * sign_p * magnitude
            self.engines[tk].state.hedge_bias = max(-1.0, min(1.0, bias))

    async def check_limits(self):
        if REAL_GROSS_LIMIT <= 0: return
        now = time.time(); wc_ratio = self.wc_gross_ratio(); sv_ratio = self.server_gross_ratio()
        if wc_ratio > GROSS_CANCEL_THRESH and not self.emergency_active: await self._cancel_building_side()
        if sv_ratio > GROSS_EMERGENCY:
            if self._can_emergency(now): await self._cancel_all_and_reduce(sv_ratio)
        if self.emergency_active and sv_ratio < GROSS_RESUME_FULL: self.emergency_active = False
    def _can_emergency(self, now):
        if now - self.last_emergency_time < EMERGENCY_COOLDOWN_S: return False
        return len([t for t in self.emergency_times if now - t < 30.0]) < MAX_EMERGENCY_PER_30
    async def _cancel_building_side(self):
        cancelled = 0
        for tk, e in self.engines.items():
            pos = e.state.server_position; tk_limit = TICKER_MAX.get(tk, 1)
            occ = abs(pos) / tk_limit if tk_limit > 0 else 0
            if occ > 0.6:
                if pos > 0 and e.state.active_buy_id:
                    try:
                        await RL.acquire()
                        async with self.session.delete(url_del(e.state.active_buy_id), timeout=0.10) as r: await r.read()
                        e.state.active_buy_id = None; e.state.pending_buy_qty = 0; cancelled += 1
                    except: pass
                elif pos < 0 and e.state.active_sell_id:
                    try:
                        await RL.acquire()
                        async with self.session.delete(url_del(e.state.active_sell_id), timeout=0.10) as r: await r.read()
                        e.state.active_sell_id = None; e.state.pending_sell_qty = 0; cancelled += 1
                    except: pass
        if cancelled > 0: self.cancel_events += 1
    async def _cancel_all_and_reduce(self, sv_ratio):
        now = time.time(); self.emergency_active = True; self.emergency_count += 1
        self.last_emergency_time = now; self.emergency_times.append(now); self.full_cancel_events += 1
        print(f"🚨 CANCEL #{self.emergency_count}: {self.gross_server()}/{REAL_GROSS_LIMIT} ({sv_ratio:.0%}) → reducing-only mode")
        try:
            await RL.acquire()
            async with self.session.post(URL_CXL, timeout=1.0) as r: await r.read()
        except: pass
        for e in self.engines.values(): e.state.reset_orders()

    def print_status(self):
        now = time.time()
        if now - self.last_status < 2.0: return
        print("\033[2J\033[H")
        gs = self.gross_server(); gsh = self.gross_shadow(); gwc = self.gross_worst_case()
        svr = gs / REAL_GROSS_LIMIT if REAL_GROSS_LIMIT > 0 else 0
        wcr = gwc / REAL_GROSS_LIMIT if REAL_GROSS_LIMIT > 0 else 0
        hc = "🟢" if svr < GROSS_THROTTLE_START else "🟡" if svr < GROSS_THROTTLE_HARD else "🟠" if svr < GROSS_CANCEL_THRESH else "🔴"
        mode = " 🔧REDUCE" if self.emergency_active else ""
        phase = self.day_phase(); ttb = self.ticks_to_boundary()
        if phase == 'flatten': mode += f" 🌙FLAT(ttb={ttb})"
        elif phase == 'ramp_down': mode += f" 📉RAMP(ttb={ttb})"
        elif phase == 'pause': mode += " ⏸️PAUSE"
        day_num = (self.current_tick // DAY_LENGTH) + 1; stats = f" day={day_num}"
        if self.gate_blocks > 0: stats += f" gate={self.gate_blocks}"
        if self.cancel_events > 0: stats += f" cxl={self.cancel_events}"
        if self.full_cancel_events > 0: stats += f" emg={self.full_cancel_events}"
        print(f"t={self.current_tick:>3} | ${self.current_pnl:>10.2f} | {hc} sv={gs} sh={gsh} wc={gwc}/{REAL_GROSS_LIMIT}({wcr:.0%}) | API:{RL.rate()}/s{mode}{stats}")
        for t, e in self.engines.items():
            s = e.state; ab = ("B" if s.active_buy_id else ".") + ("S" if s.active_sell_id else ".")
            lim = TICKER_MAX.get(t, 0); occ = abs(s.shadow_position) / lim * 100 if lim > 0 else 0
            fp = ""
            pend = f" pend=+{s.pending_buy_qty}/-{s.pending_sell_qty}" if s.pending_buy_qty > 0 or s.pending_sell_qty > 0 else ""
            cs = f" xs={s.cross_signal:>+.2f}" if abs(s.cross_signal) > 0.1 else ""
            hb = f" hb={s.hedge_bias:>+.2f}" if abs(s.hedge_bias) > 0.1 else ""
            lock = " 🔒" if occ >= 95 else ""
            sv_sh = f"{s.server_position}/{s.shadow_position}" if s.server_position != s.shadow_position else f"{s.server_position}"
            print(f"  {t}({int(TICKER_ALLOC[t]*100):>2}%) {sv_sh:>12}/{lim}({occ:>3.0f}%) {s.best_bid:.2f}/{s.best_ask:.2f} {ab}{lock}{fp}{pend}{cs}{hb}")
        self.last_status = now

    async def endgame(self):
        self.endgame_active = True; print(f"🎯 ENDGAME t={self.current_tick}")
        try:
            await RL.acquire()
            async with self.session.post(URL_CXL, timeout=1.0) as r: await r.read()
        except: pass
        for tk, e in self.engines.items():
            pos = e.state.server_position
            if abs(pos) > 50:
                act = 'SELL' if pos > 0 else 'BUY'
                pr = round((e.state.best_bid - 0.03) if act == 'SELL' else (e.state.best_ask + 0.03), 2)
                try:
                    await RL.acquire()
                    async with self.session.post(URL_ORD, params={'ticker': tk, 'type': 'LIMIT', 'action': act, 'quantity': abs(pos), 'price': pr}, timeout=1.0) as r: await r.read()
                except: pass
        await asyncio.sleep(0.3); await self._fetch_securities()
        for tk, e in self.engines.items():
            pos = e.state.server_position
            if abs(pos) > 50:
                act = 'SELL' if pos > 0 else 'BUY'
                try:
                    await RL.acquire()
                    async with self.session.post(URL_ORD, params={'ticker': tk, 'type': 'MARKET', 'action': act, 'quantity': abs(pos)}, timeout=1.0) as r: await r.read()
                except: pass
        await asyncio.sleep(0.5); global SHUTDOWN; SHUTDOWN = True

    async def run(self):
        global SHUTDOWN
        alloc_str = " | ".join(f"{tk}={TICKER_MAX[tk]}" for tk in TICKERS)
        print(f"✅ Trading | REAL={REAL_GROSS_LIMIT} | {alloc_str}")
        print(f"📅 Day boundaries: {DAY_BOUNDARIES} | Flatten lead: {FLATTEN_LEAD} ticks")
        while not SHUTDOWN:
            t0 = time.monotonic()
            if self.cycles % 100 == 0: self.current_pnl = self.pnl()
            await self.fetch_all()
            phase = self.day_phase()
            if phase == 'flatten':
                await self.flatten_for_day_end()
                if self.next_boundary() == 300 and self.ticks_to_boundary() <= 1:
                    await self._fetch_securities()
                    for tk, e in self.engines.items():
                        pos = e.state.server_position
                        if abs(pos) > 10:
                            act = 'SELL' if pos > 0 else 'BUY'
                            try:
                                await RL.acquire()
                                async with self.session.post(URL_ORD, params={'ticker': tk, 'type': 'MARKET', 'action': act, 'quantity': abs(pos)}, timeout=1.0) as r: await r.read()
                            except: pass
                    self.endgame_active = True; print(f"🎯 ENDGAME t={self.current_tick}")
                    await asyncio.sleep(0.5); SHUTDOWN = True; break
            elif phase == 'pause':
                await self.day_pause()
            elif phase == 'ramp_down':
                self.resume_from_pause()
                await self.check_limits()
                sv_ratio = self.server_gross_ratio()
                ttb = self.ticks_to_boundary()
                ramp_range = FLATTEN_RAMP_LEAD - FLATTEN_LEAD
                scale = 0.3 + 0.7 * max(0.0, (ttb - FLATTEN_LEAD) / ramp_range) if ramp_range > 0 else 0.3
                for e in self.engines.values():
                    if await e.run_cycle(self.session, sv_ratio, reducing_only=self.emergency_active, boundary_scale=scale):
                        self.last_trading_activity = time.time()
            else:
                self.resume_from_pause()
                await self.check_limits()
                sv_ratio = self.server_gross_ratio()
                for e in self.engines.values():
                    if await e.run_cycle(self.session, sv_ratio, reducing_only=self.emergency_active):
                        self.last_trading_activity = time.time()
            if self.cycles % 200 == 0: self.print_status()
            self.cycles += 1
            sl = max(0, FAST_CYCLE_MS - (time.monotonic() - t0))
            if sl > 0: await asyncio.sleep(sl)

    async def cleanup(self):
        print("🛑 SHUTTING DOWN")
        try:
            await RL.acquire()
            async with self.session.post(URL_CXL, timeout=1.0) as r: await r.read()
        except: pass
        if self.session: await self.session.close()
        if sys.platform == 'win32':
            try: import ctypes; ctypes.windll.winmm.timeEndPeriod(1)
            except: pass
        gc.enable()
        print(f"📈 Emergencies: {self.emergency_count} | Gate: {self.gate_blocks} | CXL: {self.cancel_events} | PNL: ${self.current_pnl:,.2f}")
        print("✅ Done")

def _sig(signum, frame): global SHUTDOWN; SHUTDOWN = True; print("\n🛑 Signal")

async def main():
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
    bot = SeasonalMM()
    try:
        if not await bot.initialize(): print("❌ Init failed"); return
        await asyncio.sleep(0.5); await bot.run()
    except KeyboardInterrupt: print("\n🛑 Manual shutdown")
    except Exception as e: print(f"\n💥 Fatal: {e}"); import traceback; traceback.print_exc()
    finally: await bot.cleanup()

if __name__ == "__main__": asyncio.run(main())
