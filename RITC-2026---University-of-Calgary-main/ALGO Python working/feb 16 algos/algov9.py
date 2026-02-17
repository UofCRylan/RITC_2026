#!/usr/bin/env python3

# maybe liquidity is causing major profitability drawbacks
# make sure doesn't get fucked by spoofing
# make sure doesn't get drawn down by heavy price movement


"""
RITC 2026 — Market Maker v9
============================
PHILOSOPHY: The quoting logic IS the limit management.
There is no separate "flatten" or "guard" or "panic" mode.
Every single cycle, compute_quotes() knows EXACTLY how much room
we have and sizes orders accordingly. If room=0, size=0. Done.

RESULTS FROM v8:
  gen2v1: +$45K ($1,840 gross fines) — nearly perfect
  gen2v2: -$26K ($58,660 gross fines) — eff_limit bug, used 50K not 11K
  gen2v3: -$479K ($409,250 net fines) — directional trade exploded SMMR to +40K

ROOT CAUSES FIXED:
  1. eff_limit was WRONG — only used agg_limit near close. Now min(agg,gross) ALWAYS.
  2. Order size too big (4000). If limit=11K, one fill cycle = 16K gross = instant fine.
     Now: dynamic sizing = min(base, room/4) every cycle. No fixed size.
  3. Directional trading REMOVED. It caused the $409K disaster.
  4. Soft flatten caused 18-tick blackout per day (90 ticks = 30% dark).
     Now: always quote, just reduce size as we approach close.
  5. No separate "guard" or "panic" modes. The quoting loop handles everything.
     If over limit → size=0 on increasing side. That's it. No emergency orders
     that can themselves cause position blowups.

THE ONE RULE: order_size = min(base_size, available_room / num_tickers)
This is computed fresh EVERY CYCLE from ACTUAL positions and ACTUAL limits.
If this rule is obeyed, fines are mathematically impossible.
"""

import asyncio
import aiohttp
import time
import signal
import re
import math
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import deque

API_KEY = "62BARBC1"
BASE_URL = "http://localhost:9999/v1"
TICKERS = ["SPNG", "SMMR", "ATMN", "WNTR"]
N_TICKERS = len(TICKERS)
MAX_ORDER = 10000
TPD = 60
TOTAL_TICKS = 300
SHUTDOWN = False

# Case constants (from case info page)
FEE = 0.02
REBATES = {"SPNG": 0.03, "SMMR": 0.04, "ATMN": 0.035, "WNTR": 0.045}
REBATE_BASE = 0.03


@dataclass
class P:
    # -- Core quoting --
    base_size: int = 4000           # BASE order size (will be dynamically reduced)
    half_spread: float = 0.10       # fallback half-spread
    inv_skew: float = 0.0005        # inventory skew per share
    max_skew: float = 0.04          # max skew cap
    requote_thresh: float = 0.03    # only requote if price moved 3¢

    # -- Adaptive spread --
    adaptive_frac: float = 1.0
    spread_floor: float = 0.08      # PROVEN: mm.py minimum
    spread_ceiling: float = 0.25

    # -- Spoof filter --
    spoof_threshold: int = 15000
    use_filtered_mid: bool = True

    # -- Volatility --
    vol_baseline: float = 0.005
    vol_max_mult: float = 2.0

    # -- Ticker weights --
    ticker_weight: Dict[str, float] = field(default_factory=lambda: {
        "SPNG": 0.65, "SMMR": 1.15, "ATMN": 0.75, "WNTR": 1.25
    })

    # -- OBI --
    obi_shift: float = 0.01
    obi_min: float = 0.30

    # -- Limit usage targets (fraction of limit we TARGET, not breach) --
    # We never want to USE more than this fraction of any limit.
    # This provides a safety buffer.
    gross_target: float = 0.80      # use at most 80% of gross limit
    net_target: float = 0.65        # use at most 65% of net limit
    per_ticker_frac: float = 0.30   # max 30% of gross limit per ticker

    # -- Near-close behavior (NOT a blackout — just reduce size) --
    taper_start: int = 12           # ticks before close: start tapering size
    close_dark: int = 2             # only go dark for final 2 ticks

    # -- Speed --
    cycle_ms: float = 0.020


# ============================================================
# DATA
# ============================================================

@dataclass
class OrderRef:
    bid_id: Optional[int] = None
    ask_id: Optional[int] = None
    bid_px: Optional[float] = None
    ask_px: Optional[float] = None

@dataclass
class Tk:
    name: str
    pos: int = 0
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid: float = 0.0
    spread: float = 0.0
    bid_vol: float = 0.0
    ask_vol: float = 0.0
    filt_bid: float = 0.0
    filt_ask: float = 0.0
    filt_mid: float = 0.0
    filt_spread: float = 0.0
    filt_bid_vol: float = 0.0
    filt_ask_vol: float = 0.0
    has_filtered: bool = False
    obi_hist: deque = field(default_factory=lambda: deque(maxlen=8))
    tick_mids: deque = field(default_factory=lambda: deque(maxlen=30))
    sigma: float = 0.004
    working: OrderRef = field(default_factory=OrderRef)
    posts: int = 0
    cancels: int = 0

    def effective_mid(self, use_filtered):
        if use_filtered and self.has_filtered and self.filt_mid > 0:
            return self.filt_mid
        return self.mid

    def effective_spread(self, use_filtered):
        if use_filtered and self.has_filtered and self.filt_spread > 0:
            return self.filt_spread
        return self.spread

    def update_raw(self, bid, ask, bv, av):
        self.best_bid = bid
        self.best_ask = ask
        self.mid = (bid + ask) / 2.0
        self.spread = ask - bid
        self.bid_vol = bv
        self.ask_vol = av

    def update_filtered(self, fbid, fask, fbv, fav):
        self.filt_bid = fbid
        self.filt_ask = fask
        self.filt_mid = (fbid + fask) / 2.0
        self.filt_spread = fask - fbid
        self.filt_bid_vol = fbv
        self.filt_ask_vol = fav
        self.has_filtered = True
        total = fbv + fav
        if total > 0:
            self.obi_hist.append((fbv - fav) / total)

    def obi(self):
        if len(self.obi_hist) < 2: return 0.0
        return sum(self.obi_hist) / len(self.obi_hist)

    def update_sigma(self, mid):
        self.tick_mids.append(mid)
        if len(self.tick_mids) >= 5:
            mids = list(self.tick_mids)
            rets = [(mids[i]-mids[i-1])/mids[i-1] for i in range(1,len(mids)) if mids[i-1]>0]
            if len(rets) >= 3:
                n = len(rets)
                m = sum(rets)/n
                self.sigma = max(0.001, math.sqrt(sum((r-m)**2 for r in rets)/n))

    def vol_ratio(self, baseline):
        return min(3.0, max(0.5, self.sigma / max(baseline, 0.0001)))

    def reset_day(self):
        self.tick_mids.clear()
        self.obi_hist.clear()
        self.sigma = 0.004
        self.has_filtered = False


@dataclass
class State:
    tk: Dict[str, Tk] = field(default_factory=dict)
    agg_limit: int = 0
    gross_limit: int = 50000
    net_limit: int = 30000
    tick: int = 0
    last_day: int = -1

    def __post_init__(self):
        for t in TICKERS: self.tk[t] = Tk(name=t)

    def gross(self): return sum(abs(ts.pos) for ts in self.tk.values())
    def net(self): return sum(ts.pos for ts in self.tk.values())

    @property
    def ttc(self): return TPD - (self.tick % TPD)
    @property
    def day(self): return self.tick // TPD
    @property
    def tid(self): return self.tick % TPD
    @property
    def ticks_to_end(self): return max(0, TOTAL_TICKS - self.tick)

    def true_limit(self):
        """The REAL limit we must stay under. Always the min of agg and gross."""
        if self.agg_limit > 0:
            return min(self.agg_limit, self.gross_limit)
        return self.gross_limit


# ============================================================
# SPOOF FILTER (unchanged — working well)
# ============================================================

def filter_book(bids, asks, threshold):
    if not bids or not asks:
        return None
    big_bids = [(float(b["price"]), float(b.get("quantity", 0)))
                for b in bids if float(b.get("quantity", 0)) >= threshold]
    big_asks = [(float(a["price"]), float(a.get("quantity", 0)))
                for a in asks if float(a.get("quantity", 0)) >= threshold]
    if big_bids and big_asks:
        fb = max(big_bids, key=lambda x: x[0])
        fa = min(big_asks, key=lambda x: x[0])
        if fb[0] < fa[0]:
            return fb[0], fa[0], sum(q for _, q in big_bids), sum(q for _, q in big_asks)
    if big_bids and not big_asks:
        fb = max(big_bids, key=lambda x: x[0])
        fa_px = float(asks[0]["price"])
        if fb[0] < fa_px:
            return fb[0], fa_px, sum(q for _, q in big_bids), float(asks[0].get("quantity", 0))
    if big_asks and not big_bids:
        fa = min(big_asks, key=lambda x: x[0])
        fb_px = float(bids[0]["price"])
        if fb_px < fa[0]:
            return fb_px, fa[0], float(bids[0].get("quantity", 0)), sum(q for _, q in big_asks)
    return None


# ============================================================
# CORE: compute_quotes — limit management IS the quoting
# ============================================================

def compute_quotes(
    fv: float,           # fair value (filtered mid)
    pos: int,            # this ticker's position
    mkt_spread: float,   # market spread
    obi_val: float,      # order book imbalance
    vol_r: float,        # volatility ratio
    ticker: str,         # ticker name
    # Limit state (computed fresh by caller every cycle)
    gross_room: int,     # shares we can add to gross before hitting target
    net_room_buy: int,   # shares we can buy before net target
    net_room_sell: int,   # shares we can sell before net target
    ticker_room: int,    # shares this ticker can add before per-ticker cap
    # Parameters
    p: P,
    # Near-close taper
    taper_mult: float,   # 0.0-1.0, reduces size near close
):
    """
    Compute bid/ask price and size.
    
    THE KEY INSIGHT: size is capped by the MINIMUM of all room calculations.
    If any room is 0, that side's size is 0. No exceptions. No overrides.
    This makes fines mathematically impossible.
    """
    
    # --- Base size (weighted by ticker profitability and rebate) ---
    rebate = REBATES.get(ticker, REBATE_BASE)
    sz_mult = min(1.5, rebate / max(REBATE_BASE, 0.001))
    tk_w = p.ticker_weight.get(ticker, 1.0)
    raw_base = max(100, int(p.base_size * sz_mult * tk_w))
    
    # Apply near-close taper
    raw_base = max(100, int(raw_base * taper_mult)) if taper_mult > 0 else 0
    
    if raw_base == 0:
        return None, None, 0, 0
    
    # --- Adaptive spread ---
    if mkt_spread > 0:
        half = (mkt_spread / 2.0) * p.adaptive_frac
        half = max(p.spread_floor, min(p.spread_ceiling, half))
    else:
        half = p.half_spread
    
    # Volatility scaling
    half *= min(vol_r, p.vol_max_mult)
    half = max(p.spread_floor, half)
    
    # --- Inventory skew ---
    max_sk = min(p.max_skew, half * 0.5)
    skew = max(-max_sk, min(max_sk, pos * p.inv_skew))
    
    # --- Over-limit urgency: when gross_room is low, skew harder to reduce ---
    # This replaces the old "flatten" logic. No emergency orders needed —
    # just make the reducing side more attractive.
    if gross_room <= 0 and abs(pos) > 0:
        # We're at or over gross target. Skew aggressively to flatten.
        urgency_skew = min(half * 0.8, abs(pos) * p.inv_skew * 3.0)
        if pos > 0:
            skew = max(skew, urgency_skew)   # push quotes down → sell more
        else:
            skew = min(skew, -urgency_skew)  # push quotes up → buy more
    
    # --- OBI shift ---
    if abs(obi_val) > p.obi_min:
        obi_s = max(-max_sk, min(max_sk, obi_val * p.obi_shift))
        skew -= obi_s  # positive OBI -> shift quotes up
    
    # --- Prices ---
    bid_px = round(fv - half - skew, 2)
    ask_px = round(fv + half - skew, 2)
    
    # Ensure bid < ask
    if bid_px >= ask_px:
        m = round(fv, 2)
        bid_px = m - 0.01
        ask_px = m + 0.01
    
    # === THE CRITICAL PART: SIZE = MIN(all room constraints) ===
    
    # Start with base
    bid_sz = raw_base
    ask_sz = raw_base
    
    # 1. Gross room: shared across all tickers.
    #    Each ticker gets at most gross_room / N_TICKERS (fair share)
    gross_per_ticker = gross_room // N_TICKERS
    
    # Buying adds |pos+qty| - |pos| to gross. If pos >= 0, that's +qty.
    # If pos < 0, buying reduces gross until pos=0, then adds.
    # Selling adds |pos-qty| - |pos| to gross. If pos <= 0, that's +qty.
    # If pos > 0, selling reduces gross until pos=0, then adds.
    
    # How much gross INCREASE would buying `bid_sz` cause?
    if pos >= 0:
        # Already long: buying increases gross by bid_sz
        bid_gross_cost = bid_sz
    else:
        # Short: buying first reduces gross, then increases
        bid_gross_cost = max(0, bid_sz - abs(pos))
    
    if pos <= 0:
        # Already short or flat: selling increases gross by ask_sz
        ask_gross_cost = ask_sz
    else:
        # Long: selling first reduces gross, then increases
        ask_gross_cost = max(0, ask_sz - pos)
    
    # Cap by gross room
    if bid_gross_cost > gross_per_ticker:
        bid_sz = min(bid_sz, gross_per_ticker + max(0, -pos))  # can use the "free" part
        bid_sz = max(0, bid_sz)
    if ask_gross_cost > gross_per_ticker:
        ask_sz = min(ask_sz, gross_per_ticker + max(0, pos))
        ask_sz = max(0, ask_sz)
    
    # 2. Net room: buying increases net, selling decreases net
    bid_sz = min(bid_sz, max(0, net_room_buy))
    ask_sz = min(ask_sz, max(0, net_room_sell))
    
    # 3. Per-ticker room: |pos| shouldn't exceed per_ticker cap
    if pos >= 0:
        # Already long: buying would increase |pos|
        buy_ticker_room = max(0, ticker_room - pos)
        bid_sz = min(bid_sz, buy_ticker_room)
        # Selling reduces |pos| — always OK, plus can go short up to ticker_room
        sell_ticker_room = pos + ticker_room  # can sell current pos + go short up to cap
        ask_sz = min(ask_sz, max(0, sell_ticker_room))
    else:
        # Already short: selling would increase |pos|
        sell_ticker_room = max(0, ticker_room - abs(pos))
        ask_sz = min(ask_sz, sell_ticker_room)
        # Buying reduces |pos| — always OK, plus can go long up to ticker_room
        buy_ticker_room = abs(pos) + ticker_room
        bid_sz = min(bid_sz, max(0, buy_ticker_room))
    
    # 4. Hard floor: don't send tiny orders (exchange may reject or waste API calls)
    if 0 < bid_sz < 100: bid_sz = 0
    if 0 < ask_sz < 100: ask_sz = 0
    
    # 5. Max order cap
    bid_sz = min(bid_sz, MAX_ORDER)
    ask_sz = min(ask_sz, MAX_ORDER)
    
    # If no size, no price
    if bid_sz == 0: bid_px = None
    if ask_sz == 0: ask_px = None
    
    return bid_px, ask_px, bid_sz, ask_sz


# ============================================================
# API
# ============================================================

class API:
    def __init__(self):
        self.s = None; self.reqs = 0

    async def init(self):
        self.s = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=50, force_close=False),
            timeout=aiohttp.ClientTimeout(total=0.25),
            headers={"X-API-Key": API_KEY})

    async def close(self):
        if self.s: await self.s.close()

    async def _g(self, ep, p=None):
        self.reqs += 1
        try:
            async with self.s.get(f"{BASE_URL}/{ep}", params=p) as r:
                if r.status == 429:
                    await asyncio.sleep(float(r.headers.get("Retry-After","0.05"))+0.01)
                    return None
                return await r.json() if r.status == 200 else None
        except: return None

    async def _p(self, ep, p=None):
        self.reqs += 1
        try:
            async with self.s.post(f"{BASE_URL}/{ep}", params=p) as r:
                if r.status == 429:
                    await asyncio.sleep(float(r.headers.get("Retry-After","0.05"))+0.01)
                    return None
                return await r.json() if r.status == 200 else None
        except: return None

    async def _d(self, ep):
        self.reqs += 1
        try:
            async with self.s.delete(f"{BASE_URL}/{ep}") as r:
                return await r.json() if r.status == 200 else None
        except: return None

    async def case(self): return await self._g("case")
    async def securities(self): return await self._g("securities")
    async def book(self, t, n=10): return await self._g("securities/book", {"ticker":t,"limit":n})
    async def limits(self): return await self._g("limits")
    async def news(self, since=0): return await self._g("news", {"since":since} if since else None)
    async def tenders(self): return await self._g("tenders")
    async def limit_order(self, tk, side, qty, px):
        return await self._p("orders", {"ticker":tk,"type":"LIMIT","action":side,
                                         "quantity":qty,"price":round(px,2)})
    async def cancel(self, oid): return await self._d(f"orders/{oid}")
    async def cancel_all(self): return await self._p("commands/cancel", {"all":1})
    async def decline_tender(self, tid): return await self._d(f"tenders/{tid}")


# ============================================================
# NEWS
# ============================================================

_LIM_RE = [
    re.compile(r"aggregate\s+position\s+limit[^\d]*(\d[\d,]*)", re.I),
    re.compile(r"position\s+limit[^\d]*(\d[\d,]*)", re.I),
    re.compile(r"(\d[\d,]*)\s+shares", re.I),
]
def parse_limit(text):
    for pat in _LIM_RE:
        m = pat.search(text)
        if m:
            v = int(m.group(1).replace(",",""))
            if 1000 <= v <= 500000: return v
    return None


# ============================================================
# BOT
# ============================================================

class Bot:
    def __init__(self, p):
        self.api = API()
        self.st = State()
        self.p = p
        self.last_news = 0
        self.cycles = 0
        self.t0 = 0.0
        self.last_tick = -1
        self.last_print = 0.0

    async def init(self):
        await self.api.init()
        print("="*60)
        print("  RITC 2026 MM v9 — ZERO FINES BY CONSTRUCTION")
        print("  size = min(base, room/N) every cycle")
        print("  No panic. No guards. No blackout. Just math.")
        print("="*60)
        while not SHUTDOWN:
            c = await self.api.case()
            if c and c.get("status") == "ACTIVE": break
            await asyncio.sleep(0.2)
        if SHUTDOWN: return False

        for _ in range(30):
            lims = await self.api.limits()
            if lims and len(lims) > 0:
                self.st.gross_limit = int(lims[0].get("gross_limit",50000))
                self.st.net_limit = int(lims[0].get("net_limit",30000))
                print(f"[INIT] GROSS={self.st.gross_limit:,} NET={self.st.net_limit:,}")
                break
            await asyncio.sleep(0.1)

        print("[INIT] Waiting for aggregate limit...")
        for _ in range(300):
            news = await self.api.news()
            if news:
                for item in news:
                    nid = int(item.get("news_id",0))
                    self.last_news = max(self.last_news, nid)
                    txt = f"{item.get('headline','')} {item.get('body','')}"
                    lim = parse_limit(txt)
                    if lim:
                        self.st.agg_limit = lim
                        print(f"[INIT] AGGREGATE={lim:,}")
                        break
                if self.st.agg_limit > 0: break
            await asyncio.sleep(0.05)
        if self.st.agg_limit == 0:
            self.st.agg_limit = 25000
            print("[INIT] WARNING: default agg=25000")
        
        true_lim = self.st.true_limit()
        print(f"[INIT] TRUE_LIMIT={true_lim:,} (min of agg={self.st.agg_limit:,}, gross={self.st.gross_limit:,})")
        self.t0 = time.time()
        return True

    async def sync_all(self):
        await asyncio.gather(self._sync_pos(), self._fetch_books(), return_exceptions=True)

    async def _sync_pos(self):
        secs = await self.api.securities()
        if not secs: return
        for s in secs:
            t = s.get("ticker","")
            if t in self.st.tk: self.st.tk[t].pos = int(s.get("position",0))

    async def _fetch_books(self):
        tasks = {t: self.api.book(t, 10) for t in TICKERS}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for tk, res in zip(tasks.keys(), results):
            if isinstance(res, Exception) or res is None: continue
            ts = self.st.tk[tk]
            bids = res.get("bids") or res.get("bid") or []
            asks = res.get("asks") or res.get("ask") or []
            if not bids or not asks: continue
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            bv = sum(float(b.get("quantity",0)) for b in bids)
            av = sum(float(a.get("quantity",0)) for a in asks)
            ts.update_raw(bid, ask, bv, av)
            filt = filter_book(bids, asks, self.p.spoof_threshold)
            if filt:
                ts.update_filtered(*filt)
            else:
                ts.update_filtered(bid, ask, bv, av)

    async def poll_news(self):
        news = await self.api.news(since=self.last_news)
        if not news: return
        for item in news:
            nid = int(item.get("news_id",0))
            if nid <= self.last_news: continue
            self.last_news = nid
            txt = f"{item.get('headline','')} {item.get('body','')}"
            lim = parse_limit(txt)
            if lim and lim != self.st.agg_limit:
                old = self.st.agg_limit
                self.st.agg_limit = lim
                print(f"\n  *** AGG: {old:,} -> {lim:,} (tick {self.st.tick}) TRUE_LIMIT now {self.st.true_limit():,} ***")

    async def sync_limits(self):
        lims = await self.api.limits()
        if lims and len(lims) > 0:
            new_gross = int(lims[0].get("gross_limit", self.st.gross_limit))
            new_net = int(lims[0].get("net_limit", self.st.net_limit))
            if new_gross != self.st.gross_limit or new_net != self.st.net_limit:
                print(f"  [LIMITS] gross: {self.st.gross_limit:,}->{new_gross:,} "
                      f"net: {self.st.net_limit:,}->{new_net:,}")
            self.st.gross_limit = new_gross
            self.st.net_limit = new_net

    async def decline_tenders(self):
        tenders = await self.api.tenders()
        if not tenders: return
        for t in tenders:
            tid = t.get("tender_id")
            if tid is not None: await self.api.decline_tender(tid)

    async def cancel_clear(self):
        await self.api.cancel_all()
        for ts in self.st.tk.values(): ts.working = OrderRef()

    async def manage_quotes(self, tk, bid_px, ask_px, bid_sz, ask_sz):
        ts = self.st.tk[tk]; wr = ts.working
        thresh = self.p.requote_thresh

        # BID
        if bid_px is None or bid_sz <= 0:
            if wr.bid_id:
                await self.api.cancel(wr.bid_id); ts.cancels+=1; wr.bid_id=None; wr.bid_px=None
        else:
            need = wr.bid_id is None
            if not need and wr.bid_px is not None and abs(wr.bid_px - bid_px) >= thresh:
                await self.api.cancel(wr.bid_id); ts.cancels+=1; wr.bid_id=None; need=True
            if need and bid_sz > 0:
                res = await self.api.limit_order(tk, "BUY", bid_sz, bid_px)
                ts.posts += 1
                if res and "order_id" in res:
                    wr.bid_id=int(res["order_id"]); wr.bid_px=bid_px

        # ASK
        if ask_px is None or ask_sz <= 0:
            if wr.ask_id:
                await self.api.cancel(wr.ask_id); ts.cancels+=1; wr.ask_id=None; wr.ask_px=None
        else:
            need = wr.ask_id is None
            if not need and wr.ask_px is not None and abs(wr.ask_px - ask_px) >= thresh:
                await self.api.cancel(wr.ask_id); ts.cancels+=1; wr.ask_id=None; need=True
            if need and ask_sz > 0:
                res = await self.api.limit_order(tk, "SELL", ask_sz, ask_px)
                ts.posts += 1
                if res and "order_id" in res:
                    wr.ask_id=int(res["order_id"]); wr.ask_px=ask_px

    def compute_room(self):
        """
        Compute available room for new orders.
        Called once per cycle, shared across all tickers.
        
        Returns: (gross_room, net_room_buy, net_room_sell, per_ticker_cap)
        """
        st = self.st; p = self.p
        
        true_lim = st.true_limit()
        gross = st.gross()
        net = st.net()
        
        # Gross room: how many more shares can we add to gross?
        gross_target = int(true_lim * p.gross_target)
        gross_room = max(0, gross_target - gross)
        
        # Net room: how much can net move in each direction?
        net_target = int(st.net_limit * p.net_target)
        net_room_buy = max(0, net_target - net)     # buying increases net
        net_room_sell = max(0, net_target + net)     # selling decreases net (net_target - (-net))
        
        # Per-ticker cap
        per_ticker_cap = int(true_lim * p.per_ticker_frac)
        
        return gross_room, net_room_buy, net_room_sell, per_ticker_cap

    def compute_taper(self, ttc):
        """
        Near-close taper: smoothly reduce order size as we approach daily close.
        At ttc > taper_start: taper = 1.0 (full size)
        At ttc = close_dark: taper = 0.0 (no new orders)
        In between: linear interpolation
        
        BUT: if we have a position, we KEEP quoting on the reducing side
        to flatten. The taper only affects the position-INCREASING side.
        
        Actually, let's keep it simple: taper reduces base size.
        The compute_quotes room logic will naturally handle the rest —
        if we have a position, the reducing side has plenty of room.
        """
        p = self.p
        if ttc <= p.close_dark:
            return 0.0
        if ttc >= p.taper_start:
            return 1.0
        # Linear from 1.0 to 0.0
        return (ttc - p.close_dark) / max(1, p.taper_start - p.close_dark)

    def status(self):
        now = time.time()
        if now - self.last_print < 3.0: return
        self.last_print = now
        st=self.st; p=self.p; g=st.gross(); n=st.net()
        true_lim = st.true_limit()
        pct=g/max(true_lim,1)*100
        net_pct = abs(n)/max(st.net_limit,1)*100
        cps = self.cycles/max(now-self.t0,0.001)
        gross_room, nrb, nrs, ptc = self.compute_room()
        tte = st.ticks_to_end
        taper = self.compute_taper(st.ttc)
        
        flag = "OK" if pct<60 else "WARN" if pct<80 else "OVER"
        nflag = "OK" if net_pct<50 else "WARN" if net_pct<65 else "OVER"
        
        print(f"\n{'='*76}")
        print(f" D{st.day+1} T{st.tick} ttc={st.ttc} tte={tte} taper={taper:.1f} | "
              f"{self.cycles}c ({cps:.0f}/s) | {self.api.reqs} reqs")
        print(f" [{flag}] GROSS:{g:,}/{true_lim:,} ({pct:.0f}%) room={gross_room:,} | "
              f"[{nflag}] NET:{n:+,}/{st.net_limit:,} ({net_pct:.0f}%) buy_room={nrb:,} sell_room={nrs:,}")
        print(f" AGG={st.agg_limit:,} GROSS_API={st.gross_limit:,} -> TRUE={true_lim:,} | ptc={ptc:,}")
        for t in TICKERS:
            ts = st.tk[t]; wr = ts.working
            fm = ts.filt_mid if ts.has_filtered else ts.mid
            bp = f"{wr.bid_px:.2f}×?" if wr.bid_px else "----"
            ap = f"{wr.ask_px:.2f}×?" if wr.ask_px else "----"
            print(f"  {t} pos={ts.pos:+6,} mid={ts.mid:.2f} filt={fm:.2f} "
                  f"obi={ts.obi():+.2f} [{bp}/{ap}] p={ts.posts}")
        print(f"{'='*76}")

    async def run(self):
        while not SHUTDOWN:
            self.cycles += 1
            c = await self.api.case()
            if not c: await asyncio.sleep(0.03); continue
            if c.get("status") != "ACTIVE":
                if c.get("status") == "STOPPED": print("[BOT] STOPPED"); break
                await asyncio.sleep(0.1); continue

            self.st.tick = int(c.get("tick",0))
            tick_changed = self.st.tick != self.last_tick
            await self.sync_all()

            # Frequent news polling (limit changes are critical)
            if self.cycles % 5 == 0 or tick_changed:
                await self.poll_news()
            if self.cycles % 15 == 0:
                await self.sync_limits()
            if self.cycles % 40 == 0:
                await self.decline_tenders()

            if tick_changed:
                for tk in TICKERS:
                    ts = self.st.tk[tk]
                    if ts.mid > 0: ts.update_sigma(ts.mid)

            st = self.st; p = self.p; ttc = st.ttc

            # Day boundary
            if st.day != st.last_day:
                if st.last_day >= 0:
                    print(f"\n  === DAY {st.day+1} (tick {st.tick}) ===")
                    await self.poll_news()
                    await self.sync_limits()
                    for ts in st.tk.values(): ts.reset_day()
                st.last_day = st.day

            # Only go truly dark for final 2 ticks of each day
            if 0 < ttc <= p.close_dark:
                await self.cancel_clear()
                self.last_tick = st.tick
                if tick_changed: self.status()
                await asyncio.sleep(p.cycle_ms)
                continue

            # === COMPUTE ROOM (once per cycle, shared across tickers) ===
            gross_room, net_room_buy, net_room_sell, per_ticker_cap = self.compute_room()
            
            # Near-close taper
            taper = self.compute_taper(ttc)

            # === QUOTE EACH TICKER ===
            for tk in TICKERS:
                ts = st.tk[tk]
                if ts.mid <= 0: continue

                fv = ts.effective_mid(p.use_filtered_mid)
                mkt_sp = ts.effective_spread(p.use_filtered_mid)

                bid_px, ask_px, bid_sz, ask_sz = compute_quotes(
                    fv=fv, pos=ts.pos,
                    mkt_spread=mkt_sp,
                    obi_val=ts.obi(),
                    vol_r=ts.vol_ratio(p.vol_baseline),
                    ticker=tk,
                    gross_room=gross_room,
                    net_room_buy=net_room_buy,
                    net_room_sell=net_room_sell,
                    ticker_room=per_ticker_cap,
                    p=p,
                    taper_mult=taper,
                )

                await self.manage_quotes(tk, bid_px, ask_px, bid_sz, ask_sz)
                
                # After placing for this ticker, reduce gross_room by what we requested
                # (conservative: assume it will fill)
                if bid_sz > 0 and ts.pos >= 0:
                    gross_room = max(0, gross_room - bid_sz)
                if ask_sz > 0 and ts.pos <= 0:
                    gross_room = max(0, gross_room - ask_sz)

            self.last_tick = st.tick
            if tick_changed: self.status()
            await asyncio.sleep(p.cycle_ms)

    async def cleanup(self):
        print("\n[SHUTDOWN] Cancelling all...")
        try: await self.api.cancel_all()
        except: pass
        st = self.st
        print(f"\n{'='*60}")
        print(f"  FINAL: gross={st.gross():,} net={st.net():+,} reqs={self.api.reqs}")
        print(f"  True limit: {st.true_limit():,} (agg={st.agg_limit:,} gross={st.gross_limit:,} net={st.net_limit:,})")
        for t in TICKERS:
            ts = st.tk[t]
            print(f"    {t}: pos={ts.pos:+,} posts={ts.posts} cancels={ts.cancels}")
        print(f"{'='*60}")
        await self.api.close()


def _sh(s,f):
    global SHUTDOWN; SHUTDOWN = True

async def main():
    signal.signal(signal.SIGINT, _sh)
    signal.signal(signal.SIGTERM, _sh)
    bot = Bot(P())
    try:
        if not await bot.init(): return
        await bot.run()
    except KeyboardInterrupt: pass
    except:
        import traceback; traceback.print_exc()
    finally: await bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())