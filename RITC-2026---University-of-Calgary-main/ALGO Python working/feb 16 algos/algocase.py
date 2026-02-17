# most consistently upward trending algo I have so far


"""
RITC 2026 — Market Maker v8
============================
Based on v7 which produced:
  - preportv3: +$24,136 P&L ($85,866 realized - $61,730 net fines)
  - preportv2: -$14,724 P&L ($60,800 realized - $75,525 net fines)
  - Beautiful uptrend to ~$95K then crash at end from fines.

v8 CHANGES (surgical — don't break what works):

  1. REMOVE BLACKOUT — replace with immediate limit detection + directional trade.
     At day boundary: fetch new limit immediately. If limit changed:
       - Decreased → SHORT (others must dump) → cover 3-5 ticks later
       - Increased → LONG (others can buy more) → sell 3-5 ticks later
     If unchanged → resume MM immediately. No dead time.

  2. AIRTIGHT NET LIMIT — the $61-75K leak.
     v7's net_guard only fires when |net| > net_limit (already fined).
     v8: continuous net awareness. At 60% net: skew all quotes to reduce.
     At 80%: reduce-only on net-increasing side. At 90%: emergency cross-spread.
     At end of sim: aggressive net flattening starting 30 ticks out.

  3. DYNAMIC LIMITS — no hardcoded limit values.
     Fetch gross_limit and net_limit from /limits API every 5 cycles.
     Parse aggregate from news at every day boundary.
     The ONLY hardcoded numbers: fee=0.02, rebates per ticker.

  4. END-OF-SIMULATION FLATTEN — the fines hit hardest at tick 300.
     Starting at tick 270 (30 ticks out), begin aggressive net reduction.
     By tick 290, must be near-zero net. Use market-crossing orders.
"""

import asyncio
import aiohttp
import time
import signal
import re
import math
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque

API_KEY = "62BARBC1"
BASE_URL = "http://localhost:9999/v1"
TICKERS = ["SPNG", "SMMR", "ATMN", "WNTR"]
MAX_ORDER = 10000
TPD = 60
TOTAL_TICKS = 300
SHUTDOWN = False

# Only truly constant values — fees and rebates from case rules
FEE = 0.02
REBATES = {"SPNG": 0.03, "SMMR": 0.04, "ATMN": 0.035, "WNTR": 0.045}
REBATE_BASE = 0.03


@dataclass
class P:
    # -- Core quoting (from mm.py proven) --
    order_size: int = 4000
    half_spread: float = 0.10
    inv_skew: float = 0.0005
    max_skew: float = 0.04
    requote_thresh: float = 0.03

    # -- Adaptive spread --
    adaptive_frac: float = 1.0
    spread_floor: float = 0.08
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

    # -- Net limit enforcement (TIGHTENED from v7) --
    net_skew: float = 0.0003
    net_caution: float = 0.50      # start skewing quotes
    net_warn: float = 0.70         # aggressive skew
    net_hard: float = 0.85         # reduce-only on net-increasing side
    net_emergency: float = 0.92    # cross-spread market orders

    # -- Gross/agg limits --
    limit_warn: float = 0.70
    limit_hard: float = 0.85
    limit_extra_spread: float = 0.08
    limit_size_red: float = 0.50
    per_ticker_frac: float = 0.28

    # -- Timing --
    flatten_soft: int = 18
    flatten_hard: int = 8
    cycle_ms: float = 0.020

    # -- Directional trade on limit change --
    dir_trade_size: int = 3000     # size per ticker for directional trade
    dir_trade_ticks: int = 4       # hold directional position for N ticks

    # -- End-of-sim flatten --
    sim_flatten_start: int = 30    # ticks before 300 to start net reduction
    sim_flatten_hard: int = 15     # aggressive net flatten
    sim_flatten_panic: int = 5     # market orders to zero


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
    # Track limit changes for directional trading
    prev_agg_limit: int = 0
    limit_change_tick: int = -100  # tick when last limit change detected
    limit_change_dir: int = 0     # +1 = increased, -1 = decreased

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

    def eff_limit(self, p):
        """Dynamic effective limit — tighter near close."""
        if self.ttc <= p.flatten_soft:
            return min(self.agg_limit, self.gross_limit) if self.agg_limit > 0 else self.gross_limit
        return self.gross_limit


# ============================================================
# SPOOF FILTER (unchanged from v7 — working well)
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
        fbv = sum(q for _, q in big_bids)
        fav = sum(q for _, q in big_asks)
        if fb[0] < fa[0]:
            return fb[0], fa[0], fbv, fav

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
# QUOTING — v7 logic + TIGHTENED net limit enforcement
# ============================================================

def compute_quotes(fv, pos, p, agg_ratio, ticker, mkt_spread,
                   net_pos, net_limit, eff_limit, vol_r, obi_val):
    """Core quoting with enhanced net limit awareness."""

    # Rebate-weighted sizing
    rebate = REBATES.get(ticker, REBATE_BASE)
    sz_mult = min(1.5, rebate / max(REBATE_BASE, 0.001))
    tk_w = p.ticker_weight.get(ticker, 1.0)
    base_sz = max(100, int(p.order_size * sz_mult * tk_w))
    bid_sz = base_sz
    ask_sz = base_sz

    # Adaptive spread from (filtered) book
    if mkt_spread > 0:
        half = (mkt_spread / 2.0) * p.adaptive_frac
        half = max(p.spread_floor, min(p.spread_ceiling, half))
    else:
        half = p.half_spread

    # Volatility scaling
    half *= min(vol_r, p.vol_max_mult)
    half = max(p.spread_floor, half)

    # Inventory skew
    raw_skew = pos * p.inv_skew
    max_sk = min(p.max_skew, half * 0.5)
    skew = max(-max_sk, min(max_sk, raw_skew))

    # === ENHANCED NET LIMIT ENFORCEMENT ===
    net_r = abs(net_pos) / max(net_limit, 1)

    # Tier 1: Caution (50%) — gentle skew to reduce net
    if net_r > p.net_caution:
        ns = net_pos * p.net_skew * (net_r / p.net_caution)
        ns = max(-max_sk, min(max_sk, ns))
        skew += ns

    # Tier 2: Warning (70%) — stronger skew + size reduction on increasing side
    if net_r > p.net_warn:
        intensity = (net_r - p.net_warn) / (p.net_hard - p.net_warn)
        intensity = min(1.0, max(0.0, intensity))
        # Additional skew
        ns2 = net_pos * p.net_skew * 3.0 * intensity
        ns2 = max(-max_sk * 2, min(max_sk * 2, ns2))
        skew += ns2
        # Size reduction on net-increasing side
        reduce = 1.0 - 0.7 * intensity
        if net_pos > 0:
            bid_sz = max(100, int(bid_sz * reduce))
        else:
            ask_sz = max(100, int(ask_sz * reduce))

    # Tier 3: Hard (85%) — reduce-only on net-increasing side
    if net_r >= p.net_hard:
        if net_pos > 0:
            bid_sz = 0  # stop buying
        else:
            ask_sz = 0  # stop selling

    # Tier 4: Emergency (92%) — both sides reduce-only
    if net_r >= p.net_emergency:
        if net_pos > 0:
            bid_sz = 0
            ask_sz = min(ask_sz, max(100, abs(pos))) if pos > 0 else ask_sz
        else:
            ask_sz = 0
            bid_sz = min(bid_sz, max(100, abs(pos))) if pos < 0 else bid_sz

    # Net room cap — never place an order that would breach net limit
    if net_limit > 0:
        net_room = max(0, net_limit - abs(net_pos))
        per_ticker_net_room = net_room // max(1, len(TICKERS))
        if net_pos >= 0:
            bid_sz = min(bid_sz, per_ticker_net_room)
        else:
            ask_sz = min(ask_sz, per_ticker_net_room)

    # OBI shift (leading indicator)
    if abs(obi_val) > p.obi_min:
        obi_s = obi_val * p.obi_shift
        obi_s = max(-max_sk, min(max_sk, obi_s))
        skew -= obi_s

    # Aggregate limit guardrails
    if agg_ratio > p.limit_warn:
        pct = min(1.0, (agg_ratio - p.limit_warn) / (1.0 - p.limit_warn))
        half += p.limit_extra_spread * pct
        s = 1.0 - p.limit_size_red * pct
        bid_sz = max(0, int(base_sz * s)) if bid_sz > 0 else 0
        ask_sz = max(0, int(base_sz * s)) if ask_sz > 0 else 0

    bid_px = round(fv - half - skew, 2)
    ask_px = round(fv + half - skew, 2)

    # Per-ticker cap
    ptm = int(eff_limit * p.per_ticker_frac)
    if ptm > 0:
        ap = abs(pos)
        if ap >= ptm:
            if pos > 0: bid_sz = 0; ask_sz = min(ask_sz, ap)
            else: ask_sz = 0; bid_sz = min(bid_sz, ap)
        elif (ptm - ap) < base_sz:
            room = max(0, ptm - ap)
            if pos > 0: bid_sz = min(bid_sz, room)
            elif pos < 0: ask_sz = min(ask_sz, room)
        if ap > ptm * 0.7 and ap < ptm:
            scale = max(0.2, 1.0 - (ap/ptm - 0.7) / 0.3)
            if pos > 0: bid_sz = max(100, int(bid_sz * scale)) if bid_sz > 0 else 0
            elif pos < 0: ask_sz = max(100, int(ask_sz * scale)) if ask_sz > 0 else 0

    # Aggregate room cap
    agg_room = max(0, eff_limit - int(agg_ratio * eff_limit))
    ptr = agg_room // 4
    if pos >= 0: bid_sz = min(bid_sz, max(0, ptr))
    else: ask_sz = min(ask_sz, max(0, ptr))

    bo, ao = bid_px, ask_px

    if agg_ratio >= 1.0: return None, None, 0, 0
    if agg_ratio >= p.limit_hard:
        if pos > 0: bo = None; bid_sz = 0; ask_sz = min(ask_sz, abs(pos))
        elif pos < 0: ao = None; ask_sz = 0; bid_sz = min(bid_sz, abs(pos))
        else: return None, None, 0, 0

    # Final net hard stop
    if net_r >= p.net_hard:
        if net_pos > 0: bo = None; bid_sz = 0
        else: ao = None; ask_sz = 0

    if bo is not None and ao is not None and bo >= ao:
        m = round(fv, 2); bo = m - 0.01; ao = m + 0.01

    return bo, ao, bid_sz, ask_sz


# ============================================================
# API (unchanged from v7)
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
    async def market_order(self, tk, side, qty):
        return await self._p("orders", {"ticker":tk,"type":"MARKET","action":side,
                                         "quantity":qty})
    async def cancel(self, oid): return await self._d(f"orders/{oid}")
    async def cancel_all(self): return await self._p("commands/cancel", {"all":1})
    async def decline_tender(self, tid): return await self._d(f"tenders/{tid}")


# ============================================================
# NEWS PARSING
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
        # Directional trade tracking
        self.dir_positions = {}  # ticker -> qty we added directionally
        self.dir_unwind_tick = -1  # tick to unwind directional trades

    async def init(self):
        await self.api.init()
        print("="*60)
        print("  RITC 2026 MM v8")
        print("  NO BLACKOUT | AIRTIGHT NET LIMITS | DIRECTIONAL ON LIMIT CHANGE")
        print("="*60)
        while not SHUTDOWN:
            c = await self.api.case()
            if c and c.get("status") == "ACTIVE": break
            await asyncio.sleep(0.2)
        if SHUTDOWN: return False

        # Fetch limits dynamically
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
                        self.st.prev_agg_limit = lim
                        print(f"[INIT] AGGREGATE={lim:,}")
                        break
                if self.st.agg_limit > 0: break
            await asyncio.sleep(0.05)
        if self.st.agg_limit == 0:
            self.st.agg_limit = 25000
            self.st.prev_agg_limit = 25000
            print("[INIT] WARNING: default agg=25000")
        self.t0 = time.time()
        return True

    # ============================================================
    # DATA SYNC
    # ============================================================

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
                self.st.prev_agg_limit = old
                self.st.agg_limit = lim
                # Detect direction of change
                if lim < old:
                    self.st.limit_change_dir = -1
                    print(f"\n  *** AGG DECREASED: {old:,} -> {lim:,} (tick {self.st.tick}) ***")
                else:
                    self.st.limit_change_dir = +1
                    print(f"\n  *** AGG INCREASED: {old:,} -> {lim:,} (tick {self.st.tick}) ***")
                self.st.limit_change_tick = self.st.tick

    async def sync_limits(self):
        """Fetch gross/net limits from API — these can change between rounds."""
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

    # ============================================================
    # ORDER MANAGEMENT
    # ============================================================

    async def cancel_clear(self):
        await self.api.cancel_all()
        for ts in self.st.tk.values(): ts.working = OrderRef()

    async def manage_quotes(self, tk, bid_px, ask_px, bid_sz, ask_sz, force_cancel=False):
        ts = self.st.tk[tk]; wr = ts.working
        thresh = self.p.requote_thresh

        if force_cancel:
            if wr.bid_id: await self.api.cancel(wr.bid_id); wr.bid_id=None; wr.bid_px=None
            if wr.ask_id: await self.api.cancel(wr.ask_id); wr.ask_id=None; wr.ask_px=None
            return

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

    # ============================================================
    # POSITION GUARDS
    # ============================================================

    async def position_guard(self):
        """Emergency: over gross limit."""
        st=self.st; gross=st.gross(); eff=st.eff_limit(self.p)
        if gross <= eff: return False
        print(f"  [POS_GUARD] gross={gross:,} > eff={eff:,}, flattening")
        await self.cancel_clear()
        excess = gross - eff
        for ts in sorted(st.tk.values(), key=lambda t: abs(t.pos), reverse=True):
            if excess<=0 or ts.pos==0: continue
            side = "SELL" if ts.pos>0 else "BUY"
            qty = min(abs(ts.pos), excess, MAX_ORDER)
            if ts.mid > 0:
                # Cross the spread to get filled
                off = ts.spread * 0.3
                px = round(ts.mid - off, 2) if side=="SELL" else round(ts.mid + off, 2)
                await self.api.limit_order(ts.name, side, qty, px)
                excess -= qty
        return True

    async def net_guard(self):
        """Emergency: over net limit — use aggressive cross-spread orders."""
        st=self.st; net=st.net()
        if abs(net) <= st.net_limit: return False
        print(f"  [NET_GUARD] net={net:+,} > limit={st.net_limit:,}, EMERGENCY FLATTEN")
        await self.cancel_clear()
        excess = abs(net) - st.net_limit
        targets = sorted(st.tk.values(), key=lambda t: t.pos, reverse=(net>0))
        for ts in targets:
            if excess <= 0: break
            if net > 0 and ts.pos <= 0: break
            if net < 0 and ts.pos >= 0: break
            side = "SELL" if ts.pos > 0 else "BUY"
            qty = min(abs(ts.pos), excess, MAX_ORDER)
            if ts.mid > 0:
                # AGGRESSIVE: cross spread significantly to ensure fill
                off = ts.spread * 0.5
                px = round(ts.mid - off, 2) if side=="SELL" else round(ts.mid + off, 2)
                await self.api.limit_order(ts.name, side, qty, px)
                excess -= qty
        return True

    async def net_emergency_flatten(self):
        """Called when net ratio > emergency threshold. Cross spread hard."""
        st = self.st; net = st.net()
        if abs(net) < st.net_limit * self.p.net_emergency:
            return
        print(f"  [NET_EMERGENCY] net={net:+,}/{st.net_limit:,} ({abs(net)/st.net_limit:.0%})")
        # Find tickers contributing most to net imbalance
        if net > 0:
            # Need to sell — find most long tickers
            targets = sorted(st.tk.values(), key=lambda t: t.pos, reverse=True)
        else:
            # Need to buy — find most short tickers
            targets = sorted(st.tk.values(), key=lambda t: t.pos)
        
        reduce_needed = abs(net) - int(st.net_limit * 0.70)  # target 70% of limit
        for ts in targets:
            if reduce_needed <= 0: break
            if net > 0 and ts.pos <= 0: break
            if net < 0 and ts.pos >= 0: break
            side = "SELL" if ts.pos > 0 else "BUY"
            qty = min(abs(ts.pos), reduce_needed, MAX_ORDER)
            if qty < 100: continue
            if ts.best_bid > 0 and ts.best_ask > 0:
                # Price aggressively to ensure fill — cross the spread
                if side == "SELL":
                    px = round(ts.best_bid - 0.01, 2)  # hit the bid
                else:
                    px = round(ts.best_ask + 0.01, 2)  # lift the ask
                await self.api.limit_order(ts.name, side, qty, px)
                reduce_needed -= qty

    # ============================================================
    # DIRECTIONAL TRADING ON LIMIT CHANGE (replaces blackout)
    # ============================================================

    async def handle_limit_change(self):
        """
        When aggregate limit changes at day boundary:
        - Decreased: others must dump → go short to capture downfall
        - Increased: others can buy more → go long to capture rally
        
        This replaces the blackout window — instead of going dark,
        we actively trade the expected price movement.
        """
        st = self.st; p = self.p
        ticks_since = st.tick - st.limit_change_tick
        
        if ticks_since > p.dir_trade_ticks + 2:
            return  # Too old, no action
        
        if ticks_since == 0:
            # Just detected — enter directional position
            await self.cancel_clear()
            direction = st.limit_change_dir  # +1 or -1
            
            # Check we have room for directional trades
            gross = st.gross()
            eff = st.eff_limit(p)
            net = st.net()
            gross_room = max(0, int(eff * 0.60) - gross)
            net_room = max(0, st.net_limit - abs(net))
            
            if gross_room < 1000 or net_room < 1000:
                print(f"  [DIR] No room for directional: gross_room={gross_room:,} net_room={net_room:,}")
                return
            
            per_ticker = min(p.dir_trade_size, gross_room // 4, net_room // 4)
            if per_ticker < 500:
                return
            
            print(f"  [DIR] Limit {'DECREASED' if direction < 0 else 'INCREASED'} → "
                  f"{'SHORT' if direction < 0 else 'LONG'} {per_ticker:,}/ticker")
            
            self.dir_positions = {}
            self.dir_unwind_tick = st.tick + p.dir_trade_ticks
            
            for tk in TICKERS:
                ts = st.tk[tk]
                if ts.best_bid <= 0 or ts.best_ask <= 0:
                    continue
                if direction < 0:
                    # SHORT: sell at best bid (cross spread to get filled)
                    px = round(ts.best_bid - 0.01, 2)
                    await self.api.limit_order(tk, "SELL", per_ticker, px)
                    self.dir_positions[tk] = -per_ticker
                else:
                    # LONG: buy at best ask (cross spread to get filled)
                    px = round(ts.best_ask + 0.01, 2)
                    await self.api.limit_order(tk, "BUY", per_ticker, px)
                    self.dir_positions[tk] = per_ticker
        
        elif ticks_since >= p.dir_trade_ticks and self.dir_positions:
            # Time to unwind
            print(f"  [DIR] Unwinding directional positions after {ticks_since} ticks")
            await self.cancel_clear()
            for tk, target_pos in self.dir_positions.items():
                ts = st.tk[tk]
                # Unwind: if we went short, buy back; if long, sell
                if target_pos < 0 and ts.pos < 0:
                    qty = min(abs(ts.pos), abs(target_pos), MAX_ORDER)
                    if qty >= 100 and ts.best_ask > 0:
                        px = round(ts.best_ask + 0.01, 2)
                        await self.api.limit_order(tk, "BUY", qty, px)
                elif target_pos > 0 and ts.pos > 0:
                    qty = min(ts.pos, target_pos, MAX_ORDER)
                    if qty >= 100 and ts.best_bid > 0:
                        px = round(ts.best_bid - 0.01, 2)
                        await self.api.limit_order(tk, "SELL", qty, px)
            self.dir_positions = {}

    # ============================================================
    # END-OF-SIMULATION FLATTEN
    # ============================================================

    async def end_of_sim_flatten(self):
        """
        Aggressive net and gross flattening as we approach tick 300.
        This is separate from daily close flatten — this is the FINAL flatten.
        """
        st = self.st; p = self.p
        tte = st.ticks_to_end  # ticks to end of simulation
        
        if tte > p.sim_flatten_start:
            return False  # Not yet
        
        gross = st.gross()
        net = st.net()
        net_r = abs(net) / max(st.net_limit, 1)
        
        # Soft phase: reduce any outsized positions
        if tte <= p.sim_flatten_start and tte > p.sim_flatten_hard:
            # Target: get net to < 50% of limit
            target_net = int(st.net_limit * 0.40)
            if abs(net) > target_net:
                urgency = 1.0 - (tte - p.sim_flatten_hard) / max(1, p.sim_flatten_start - p.sim_flatten_hard)
                # Don't cancel all — just add reducing orders alongside MM
                if net > 0:
                    targets = sorted(st.tk.values(), key=lambda t: t.pos, reverse=True)
                else:
                    targets = sorted(st.tk.values(), key=lambda t: t.pos)
                
                reduce_needed = abs(net) - target_net
                for ts in targets:
                    if reduce_needed <= 0: break
                    if net > 0 and ts.pos <= 0: break
                    if net < 0 and ts.pos >= 0: break
                    side = "SELL" if ts.pos > 0 else "BUY"
                    qty = min(abs(ts.pos), reduce_needed, MAX_ORDER, int(3000 * urgency))
                    if qty < 100: continue
                    if ts.mid > 0:
                        off = 0.02 * urgency
                        px = round(ts.mid + off, 2) if side == "SELL" else round(ts.mid - off, 2)
                        await self.api.limit_order(ts.name, side, qty, px)
                        reduce_needed -= qty
            return False  # Still allow MM alongside
        
        # Hard phase: cancel all, aggressive flatten
        if tte <= p.sim_flatten_hard and tte > p.sim_flatten_panic:
            await self.cancel_clear()
            for tk in TICKERS:
                ts = st.tk[tk]
                if ts.pos == 0: continue
                side = "SELL" if ts.pos > 0 else "BUY"
                qty = min(abs(ts.pos), MAX_ORDER)
                if qty < 100: continue
                urgency = 1.0 - (tte - p.sim_flatten_panic) / max(1, p.sim_flatten_hard - p.sim_flatten_panic)
                if ts.mid > 0:
                    off = 0.05 * urgency  # cross spread more as deadline approaches
                    px = round(ts.mid + off, 2) if side == "SELL" else round(ts.mid - off, 2)
                    # Actually for SELL we want LOWER price, for BUY HIGHER
                    if side == "SELL":
                        px = round(ts.mid - off, 2)
                    else:
                        px = round(ts.mid + off, 2)
                    await self.api.limit_order(ts.name, side, qty, px)
            return True  # Block MM
        
        # Panic phase: market-crossing orders, repeated attempts
        if tte <= p.sim_flatten_panic:
            await self.cancel_clear()
            for attempt in range(3):
                await self._sync_pos()
                net = st.net()
                gross = st.gross()
                if abs(net) < 500 and gross < 500:
                    break
                print(f"  [SIM_PANIC] attempt={attempt} tte={tte} gross={gross:,} net={net:+,}")
                for tk in TICKERS:
                    ts = st.tk[tk]
                    if ts.pos == 0: continue
                    side = "SELL" if ts.pos > 0 else "BUY"
                    qty = min(abs(ts.pos), MAX_ORDER)
                    if qty < 100: continue
                    if ts.best_bid > 0 and ts.best_ask > 0:
                        # Cross spread aggressively
                        if side == "SELL":
                            px = round(ts.best_bid - 0.02, 2)
                        else:
                            px = round(ts.best_ask + 0.02, 2)
                        await self.api.limit_order(ts.name, side, qty, px)
                await asyncio.sleep(0.05)
            return True
        
        return False

    # ============================================================
    # FLATTEN LOGIC (daily close)
    # ============================================================

    async def flatten_hard_fn(self, ttc):
        """Daily close flatten — also accounts for net limit."""
        st = self.st
        for tk in TICKERS:
            ts = st.tk[tk]
            if ts.pos == 0: continue
            side = "SELL" if ts.pos > 0 else "BUY"
            qty = min(abs(ts.pos), MAX_ORDER)
            if qty < 100: continue
            urg = 1.0 - (ttc / max(1, self.p.flatten_hard))
            off = 0.03 * (1.0 - urg)
            if ts.mid > 0:
                # Cross spread direction for flatten
                if side == "SELL":
                    px = round(ts.mid - off, 2)
                else:
                    px = round(ts.mid + off, 2)
                await self.api.limit_order(tk, side, qty, px)

    # ============================================================
    # STATUS
    # ============================================================

    def status(self):
        now = time.time()
        if now - self.last_print < 3.0: return
        self.last_print = now
        st=self.st; p=self.p; g=st.gross(); n=st.net()
        eff=st.eff_limit(p); pct=g/max(eff,1)*100
        net_pct = abs(n)/max(st.net_limit,1)*100
        cps = self.cycles/max(now-self.t0,0.001)
        flag = "OK" if pct<60 else "WARN" if pct<80 else "DANGER"
        net_flag = "OK" if net_pct<50 else "WARN" if net_pct<70 else "DANGER" if net_pct<85 else "CRIT"
        tte = st.ticks_to_end
        print(f"\n{'='*76}")
        print(f" D{st.day+1} T{st.tick} ttc={st.ttc} tte={tte} | {self.cycles}c ({cps:.0f}/s) | {self.api.reqs} reqs")
        print(f" [{flag}] GROSS:{g:,}/{eff:,} ({pct:.0f}%) | "
              f"[{net_flag}] NET:{n:+,}/{st.net_limit:,} ({net_pct:.0f}%) | AGG={st.agg_limit:,}")
        for t in TICKERS:
            ts = st.tk[t]; wr = ts.working
            fm = ts.filt_mid if ts.has_filtered else ts.mid
            bp = f"{wr.bid_px:.2f}" if wr.bid_px else "----"
            ap = f"{wr.ask_px:.2f}" if wr.ask_px else "----"
            print(f"  {t} pos={ts.pos:+6,} mid={ts.mid:.2f} filt={fm:.2f} "
                  f"obi={ts.obi():+.2f} [{bp}/{ap}] p={ts.posts}")
        if self.dir_positions:
            print(f"  [DIR] active positions: {self.dir_positions}")
        print(f"{'='*76}")

    # ============================================================
    # MAIN LOOP
    # ============================================================

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

            # Poll news frequently — especially at day boundaries
            if self.cycles % 5 == 0 or tick_changed:
                await self.poll_news()
            # Sync limits from API regularly
            if self.cycles % 15 == 0:
                await self.sync_limits()
            if self.cycles % 40 == 0:
                await self.decline_tenders()

            if tick_changed:
                for tk in TICKERS:
                    ts = self.st.tk[tk]
                    if ts.mid > 0: ts.update_sigma(ts.mid)

            st = self.st; p = self.p; ttc = st.ttc; tid = st.tid
            gross = st.gross(); net_pos = st.net()
            eff = st.eff_limit(p)
            agg_r = gross / max(eff, 1)
            net_r = abs(net_pos) / max(st.net_limit, 1)
            over = gross > eff

            # === END-OF-SIMULATION FLATTEN (highest priority) ===
            if await self.end_of_sim_flatten():
                self.last_tick = st.tick
                if tick_changed: self.status()
                await asyncio.sleep(p.cycle_ms)
                continue

            # === DAY BOUNDARY — fetch new limits, detect changes ===
            if st.day != st.last_day:
                if st.last_day >= 0:
                    print(f"\n  === DAY {st.day+1} (tick {st.tick}) ===")
                    # Fetch news immediately at day boundary
                    await self.poll_news()
                    await self.sync_limits()
                    # Reset per-day tracking
                    for ts in st.tk.values(): ts.reset_day()
                    # DON'T cancel all — we might want to keep positions
                    # for directional trading
                st.last_day = st.day

            # === DIRECTIONAL TRADING ON LIMIT CHANGE (replaces blackout) ===
            ticks_since_change = st.tick - st.limit_change_tick
            if 0 <= ticks_since_change <= p.dir_trade_ticks + 2:
                await self.handle_limit_change()
                # During directional phase, skip normal MM for first 2 ticks
                if ticks_since_change < 2:
                    self.last_tick = st.tick
                    if tick_changed: self.status()
                    await asyncio.sleep(p.cycle_ms)
                    continue
                # After 2 ticks, resume MM alongside directional

            # === PRE-CLOSE DARK (only last 2 ticks of each day) ===
            if 0 < ttc <= 2:
                await self.cancel_clear()
                self.last_tick = st.tick
                if tick_changed: self.status()
                await asyncio.sleep(p.cycle_ms)
                continue

            # === EMERGENCY GUARDS ===
            if await self.position_guard():
                self.last_tick = st.tick; await asyncio.sleep(p.cycle_ms); continue
            if await self.net_guard():
                self.last_tick = st.tick; await asyncio.sleep(p.cycle_ms); continue

            # === NET EMERGENCY — proactive, before it becomes a violation ===
            if net_r >= p.net_emergency:
                await self.net_emergency_flatten()

            # === HARD FLATTEN (daily close) ===
            if p.flatten_hard >= ttc > 2:
                await self.cancel_clear()
                await self.flatten_hard_fn(ttc)
                self.last_tick = st.tick
                if tick_changed: self.status()
                await asyncio.sleep(p.cycle_ms)
                continue

            # === SOFT FLATTEN (approaching daily close) ===
            is_flat = ttc <= p.flatten_soft

            # === CORE MARKET MAKING ===
            for tk in TICKERS:
                ts = st.tk[tk]
                if ts.mid <= 0: continue

                if is_flat and ttc > p.flatten_hard:
                    # Near close: flatten positions
                    urg = 1.0 - (ttc - p.flatten_hard)/max(1, p.flatten_soft - p.flatten_hard)
                    off = 0.03 * (1.0 - urg)
                    fsz = min(abs(ts.pos), p.order_size) if ts.pos != 0 else 0
                    if ts.pos > 0:
                        # Sell to flatten — price below mid to get filled
                        await self.manage_quotes(tk, None, round(ts.mid - off, 2), 0, fsz, over)
                    elif ts.pos < 0:
                        # Buy to flatten — price above mid to get filled
                        await self.manage_quotes(tk, round(ts.mid + off, 2), None, fsz, 0, over)
                    else:
                        await self.manage_quotes(tk, None, None, 0, 0, over)
                else:
                    # Normal market making
                    fv = ts.effective_mid(p.use_filtered_mid)
                    mkt_sp = ts.effective_spread(p.use_filtered_mid)

                    bid_px, ask_px, bid_sz, ask_sz = compute_quotes(
                        fv=fv, pos=ts.pos, p=p,
                        agg_ratio=agg_r, ticker=tk,
                        mkt_spread=mkt_sp,
                        net_pos=net_pos, net_limit=st.net_limit,
                        eff_limit=eff,
                        vol_r=ts.vol_ratio(p.vol_baseline),
                        obi_val=ts.obi(),
                    )

                    # End-of-day taper
                    if ttc <= 20:
                        taper = max(0.2, ttc/20.0)
                        if bid_sz > 0: bid_sz = max(100, int(bid_sz * taper))
                        if ask_sz > 0: ask_sz = max(100, int(ask_sz * taper))

                    await self.manage_quotes(tk, bid_px, ask_px, bid_sz, ask_sz, over)

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
        print(f"  Agg limit: {st.agg_limit:,} | Gross limit: {st.gross_limit:,} | Net limit: {st.net_limit:,}")
        for t in TICKERS:
            ts = st.tk[t]
            print(f"    {t}: pos={ts.pos:+,} posts={ts.posts} cancels={ts.cancels}")
        print(f"{'='*60}")
        await self.api.close()


# ============================================================
# ENTRY
# ============================================================

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