#!/usr/bin/env python3
"""
RITC 2026 — Market Maker v7
============================
STRATEGY: mm.py exact logic (proven $75K) + async speed + spoof filter.

mm.py makes $75K but:
  1. $15K fines (limit management timing)
  2. Loses money for first 100 ticks (regime changes)
  3. 200ms cycle = 5 cycles/sec (slow)

v7 fixes:
  1. v5's zero-fine limit management (proven $0 fines)
  2. 5-tick blackout per day (avoids regime change losses)
  3. 20ms cycle = 50 cycles/sec (10x faster)
  4. SPOOF FILTER: use only orders > SPOOF_THRESHOLD for true mid-price
     Max order per trader = 10,000. Orders >> 10K are from exchange bots = REAL.
  5. Inventory skew in PRICE UNITS not returns (actually moves quotes)
  6. Adaptive spread from FILTERED book (not spoofed book)
  7. Weight WNTR/SMMR higher (consistently profitable tickers)
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
SHUTDOWN = False


@dataclass
class P:
    # -- Core quoting (from mm.py proven $75K) --
    order_size: int = 4000
    half_spread: float = 0.10       # fallback
    inv_skew: float = 0.0005        # $/share inventory skew
    max_skew: float = 0.04
    requote_thresh: float = 0.03    # only requote on 3¢ move

    # -- Adaptive spread (from mm.py) --
    adaptive_frac: float = 1.0      # fraction of market spread to match
    spread_floor: float = 0.08      # CRITICAL: mm.py's proven minimum
    spread_ceiling: float = 0.25

    # -- Spoof filter --
    spoof_threshold: int = 15000    # orders > this are likely real (exchange bots)
    use_filtered_mid: bool = True   # use spoof-filtered mid for fair value

    # -- Volatility spread scaling --
    vol_baseline: float = 0.005
    vol_max_mult: float = 2.0

    # -- Ticker weights (from P&L data) --
    ticker_weight: Dict[str, float] = field(default_factory=lambda: {
        "SPNG": 0.65, "SMMR": 1.15, "ATMN": 0.75, "WNTR": 1.25
    })
    rebates: Dict[str, float] = field(default_factory=lambda: {
        "SPNG": 0.03, "SMMR": 0.04, "ATMN": 0.035, "WNTR": 0.045
    })
    rebate_base: float = 0.03

    # -- OBI --
    obi_shift: float = 0.01        # max FV shift from book imbalance
    obi_min: float = 0.30          # minimum |OBI| to act

    # -- Net skew --
    net_skew: float = 0.0003
    net_warn: float = 0.50
    net_hard: float = 0.85

    # -- Limits (v5 proven zero fines) --
    limit_warn: float = 0.70
    limit_hard: float = 0.85
    limit_extra_spread: float = 0.08
    limit_size_red: float = 0.50
    per_ticker_frac: float = 0.28

    # -- Timing --
    blackout: int = 5
    flatten_soft: int = 18
    flatten_hard: int = 8
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
    # Raw book
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid: float = 0.0
    spread: float = 0.0
    bid_vol: float = 0.0
    ask_vol: float = 0.0
    # Filtered book (spoof-free)
    filt_bid: float = 0.0
    filt_ask: float = 0.0
    filt_mid: float = 0.0
    filt_spread: float = 0.0
    filt_bid_vol: float = 0.0
    filt_ask_vol: float = 0.0
    has_filtered: bool = False
    # OBI from filtered book
    obi_hist: deque = field(default_factory=lambda: deque(maxlen=8))
    # Realized vol
    tick_mids: deque = field(default_factory=lambda: deque(maxlen=30))
    sigma: float = 0.004
    # Working orders
    working: OrderRef = field(default_factory=OrderRef)
    posts: int = 0
    cancels: int = 0

    def effective_mid(self, use_filtered):
        """Use filtered mid if available and enabled."""
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
        # OBI from filtered book
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

    def eff_limit(self, p):
        if self.ttc <= p.flatten_soft:
            return min(self.agg_limit, self.gross_limit) if self.agg_limit > 0 else self.gross_limit
        return self.gross_limit


# ============================================================
# SPOOF FILTER
# ============================================================

def filter_book(bids: list, asks: list, threshold: int):
    """
    Filter orderbook to find "real" orders.
    
    Logic: Max order per competitor = 10,000 shares.
    Orders with quantity >> 10K at a single price level are from
    exchange market-making bots = REAL liquidity, not spoofs.
    
    We look at ALL orders and compute a "real" best bid/ask by
    weighting large orders more heavily. Small orders near the
    top that disagree with large orders deeper in the book are
    likely spoofs trying to manipulate our mid-price.
    
    Returns: (filtered_bid, filtered_ask, filtered_bid_vol, filtered_ask_vol)
             or None if insufficient data.
    """
    if not bids or not asks:
        return None

    # Find the best bid/ask from "big" orders only
    big_bids = [(float(b["price"]), float(b.get("quantity", 0)))
                for b in bids if float(b.get("quantity", 0)) >= threshold]
    big_asks = [(float(a["price"]), float(a.get("quantity", 0)))
                for a in asks if float(a.get("quantity", 0)) >= threshold]

    # If we found big orders, use those as the "true" market
    if big_bids and big_asks:
        fb = max(big_bids, key=lambda x: x[0])
        fa = min(big_asks, key=lambda x: x[0])
        fbv = sum(q for _, q in big_bids)
        fav = sum(q for _, q in big_asks)
        # Sanity: filtered bid must be < filtered ask
        if fb[0] < fa[0]:
            return fb[0], fa[0], fbv, fav

    # If only one side has big orders, use that side + raw other side
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

    # No big orders found — fall back to raw top of book
    return None


# ============================================================
# QUOTING: mm.py logic + spoof filter + proper inventory control
# ============================================================

def compute_quotes(fv, pos, p, agg_ratio, ticker, mkt_spread,
                   net_pos, net_limit, eff_limit, vol_r, obi_val):
    """mm.py proven quoting logic, adapted with spoof-filtered inputs."""

    # --- Rebate-weighted sizing ---
    rebate = p.rebates.get(ticker, p.rebate_base)
    sz_mult = min(1.5, rebate / max(p.rebate_base, 0.001))
    # Also apply ticker profitability weight
    tk_w = p.ticker_weight.get(ticker, 1.0)
    base_sz = max(100, int(p.order_size * sz_mult * tk_w))
    bid_sz = base_sz
    ask_sz = base_sz

    # --- Adaptive spread from (filtered) book ---
    if mkt_spread > 0:
        half = (mkt_spread / 2.0) * p.adaptive_frac
        half = max(p.spread_floor, min(p.spread_ceiling, half))
    else:
        half = p.half_spread

    # --- Volatility scaling ---
    half *= min(vol_r, p.vol_max_mult)

    # Floor
    half = max(p.spread_floor, half)

    # --- Inventory skew (linear, from mm.py) ---
    raw_skew = pos * p.inv_skew
    max_sk = min(p.max_skew, half * 0.5)
    skew = max(-max_sk, min(max_sk, raw_skew))

    # --- Net skew ---
    if abs(net_pos) > net_limit * p.net_warn:
        ns = net_pos * p.net_skew
        ns = max(-max_sk, min(max_sk, ns))
        skew += ns

    # --- OBI shift (leading indicator) ---
    if abs(obi_val) > p.obi_min:
        obi_s = obi_val * p.obi_shift
        obi_s = max(-max_sk, min(max_sk, obi_s))
        skew -= obi_s  # shift TOWARD pressure (buy pressure -> lower bid, higher ask... wait)
        # Actually: positive OBI = buy pressure = price likely going up
        # So we should shift FV UP = add to fair value
        # But skew is subtracted from bid and ask symmetrically
        # To shift FV up: decrease skew (move both quotes up)
        # skew -= obi_s means: positive OBI -> negative adjustment to skew -> quotes move UP
        # This is correct.

    # --- Net limit size guards ---
    net_r = abs(net_pos) / max(net_limit, 1)
    if net_r >= p.net_hard:
        if net_pos > 0: bid_sz = 0
        else: ask_sz = 0
    if net_limit > 0:
        nr = max(0, net_limit - abs(net_pos))
        ptnr = nr // 4
        if net_pos >= 0: bid_sz = min(bid_sz, max(0, ptnr))
        else: ask_sz = min(ask_sz, max(0, ptnr))

    # --- Aggregate limit guardrails ---
    if agg_ratio > p.limit_warn:
        pct = min(1.0, (agg_ratio - p.limit_warn) / (1.0 - p.limit_warn))
        half += p.limit_extra_spread * pct
        s = 1.0 - p.limit_size_red * pct
        bid_sz = max(100, int(base_sz * s))
        ask_sz = max(100, int(base_sz * s))

    bid_px = round(fv - half - skew, 2)
    ask_px = round(fv + half - skew, 2)

    # Per-ticker cap (exact mm.py logic)
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
            if pos > 0: bid_sz = max(100, int(bid_sz * scale))
            elif pos < 0: ask_sz = max(100, int(ask_sz * scale))

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

    if net_r >= p.net_hard:
        if net_pos > 0: bo = None; bid_sz = 0
        else: ao = None; ask_sz = 0

    if bo is not None and ao is not None and bo >= ao:
        m = round(fv, 2); bo = m - 0.01; ao = m + 0.01

    return bo, ao, bid_sz, ask_sz


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
        print("  RITC 2026 MM v7 — mm.py LOGIC + SPOOF FILTER + ASYNC")
        print("  $0.08 floor | 5-tick blackout | filtered book OBI")
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
        """Fetch deeper books (10 levels) for spoof filtering."""
        tasks = {t: self.api.book(t, 10) for t in TICKERS}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for tk, res in zip(tasks.keys(), results):
            if isinstance(res, Exception) or res is None: continue
            ts = self.st.tk[tk]
            bids = res.get("bids") or res.get("bid") or []
            asks = res.get("asks") or res.get("ask") or []
            if not bids or not asks: continue

            # Raw top of book
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            bv = sum(float(b.get("quantity",0)) for b in bids)
            av = sum(float(a.get("quantity",0)) for a in asks)
            ts.update_raw(bid, ask, bv, av)

            # Spoof-filtered book
            filt = filter_book(bids, asks, self.p.spoof_threshold)
            if filt:
                ts.update_filtered(*filt)
            else:
                # No big orders found — use raw but still update OBI
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
                print(f"\n  *** AGG: {old:,} -> {lim:,} (tick {self.st.tick}) ***")

    async def sync_limits(self):
        lims = await self.api.limits()
        if lims and len(lims) > 0:
            self.st.gross_limit = int(lims[0].get("gross_limit", self.st.gross_limit))
            self.st.net_limit = int(lims[0].get("net_limit", self.st.net_limit))

    async def decline_tenders(self):
        tenders = await self.api.tenders()
        if not tenders: return
        for t in tenders:
            tid = t.get("tender_id")
            if tid is not None: await self.api.decline_tender(tid)

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

    async def position_guard(self):
        st=self.st; gross=st.gross(); eff=st.eff_limit(self.p)
        if gross <= eff: return False
        await self.cancel_clear()
        excess = gross - eff
        for ts in sorted(st.tk.values(), key=lambda t: abs(t.pos), reverse=True):
            if excess<=0 or ts.pos==0: continue
            side = "SELL" if ts.pos>0 else "BUY"
            qty = min(abs(ts.pos), excess, MAX_ORDER)
            if ts.mid > 0:
                off = ts.spread*0.05
                px = round(ts.mid+off,2) if side=="SELL" else round(ts.mid-off,2)
                await self.api.limit_order(ts.name, side, qty, px)
                excess -= qty
        return True

    async def net_guard(self):
        st=self.st; net=st.net()
        if abs(net) <= st.net_limit: return False
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
                off = ts.spread*0.05
                px = round(ts.mid+off,2) if side=="SELL" else round(ts.mid-off,2)
                await self.api.limit_order(ts.name, side, qty, px)
                excess -= qty
        return True

    async def flatten_hard_fn(self, ttc):
        for tk in TICKERS:
            ts = self.st.tk[tk]
            if ts.pos == 0: continue
            side = "SELL" if ts.pos>0 else "BUY"
            qty = min(abs(ts.pos), MAX_ORDER)
            if qty < self.p.min_size if hasattr(self.p,'min_size') else qty < 100: continue
            urg = 1.0 - (ttc / max(1, self.p.flatten_hard))
            off = 0.03 * (1.0 - urg)
            if ts.mid > 0:
                px = round(ts.mid+off,2) if side=="SELL" else round(ts.mid-off,2)
                await self.api.limit_order(tk, side, qty, px)

    def status(self):
        now = time.time()
        if now - self.last_print < 3.0: return
        self.last_print = now
        st=self.st; p=self.p; g=st.gross(); n=st.net()
        eff=st.eff_limit(p); pct=g/max(eff,1)*100
        cps = self.cycles/max(now-self.t0,0.001)
        flag = "OK" if pct<60 else "WARN" if pct<80 else "DANGER"
        print(f"\n{'='*72}")
        print(f" D{st.day+1} T{st.tick} ttc={st.ttc} | {self.cycles}c ({cps:.0f}/s) | {self.api.reqs} reqs")
        print(f" [{flag}] GROSS:{g:,}/{eff:,} ({pct:.0f}%) | NET:{n:+,}/{st.net_limit:,} | AGG={st.agg_limit:,}")
        for t in TICKERS:
            ts = st.tk[t]; wr = ts.working
            fm = ts.filt_mid if ts.has_filtered else ts.mid
            delta = fm - ts.mid if ts.has_filtered else 0
            bp = f"{wr.bid_px:.2f}" if wr.bid_px else "----"
            ap = f"{wr.ask_px:.2f}" if wr.ask_px else "----"
            print(f"  {t} pos={ts.pos:+6,} mid={ts.mid:.2f} filt={fm:.2f}({delta:+.3f}) "
                  f"obi={ts.obi():+.2f} [{bp}/{ap}] p={ts.posts}")
        print(f"{'='*72}")

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

            if self.cycles % 10 == 0: await self.poll_news()
            if self.cycles % 30 == 0: await self.sync_limits()
            if self.cycles % 40 == 0: await self.decline_tenders()

            if tick_changed:
                for tk in TICKERS:
                    ts = self.st.tk[tk]
                    if ts.mid > 0: ts.update_sigma(ts.mid)

            st=self.st; p=self.p; ttc=st.ttc; tid=st.tid
            gross=st.gross(); net_pos=st.net()
            eff=st.eff_limit(p); agg_r=gross/max(eff,1)
            over = gross > eff

            # Day boundary
            if st.day != st.last_day:
                if st.last_day >= 0:
                    print(f"\n  === DAY {st.day+1} (tick {st.tick}) ===")
                    await self.cancel_clear()
                    for ts in st.tk.values(): ts.reset_day()
                st.last_day = st.day

            # Pre-close dark
            if 0 < ttc <= 2:
                await self.cancel_clear()
                self.last_tick = st.tick
                if tick_changed: self.status()
                await asyncio.sleep(p.cycle_ms); continue

            # Guards
            if await self.position_guard():
                self.last_tick = st.tick; await asyncio.sleep(p.cycle_ms); continue
            if await self.net_guard():
                self.last_tick = st.tick; await asyncio.sleep(p.cycle_ms); continue

            # Hard flatten
            if p.flatten_hard >= ttc > 2:
                await self.cancel_clear()
                await self.flatten_hard_fn(ttc)
                self.last_tick = st.tick
                if tick_changed: self.status()
                await asyncio.sleep(p.cycle_ms); continue

            # Blackout
            if tid < p.blackout:
                await self.cancel_clear()
                if tick_changed:
                    parts = [f"{t}={st.tk[t].mid:.2f}" for t in TICKERS if st.tk[t].mid > 0]
                    print(f"  [BLACKOUT] {tid}/{p.blackout}: {' '.join(parts)}")
                self.last_tick = st.tick
                await asyncio.sleep(p.cycle_ms); continue

            # Soft flatten
            is_flat = ttc <= p.flatten_soft

            for tk in TICKERS:
                ts = st.tk[tk]
                if ts.mid <= 0: continue

                if is_flat and ttc > p.flatten_hard:
                    urg = 1.0 - (ttc - p.flatten_hard)/max(1, p.flatten_soft - p.flatten_hard)
                    off = 0.03 * (1.0 - urg)
                    fsz = min(abs(ts.pos), p.order_size) if ts.pos != 0 else 0
                    if ts.pos > 0:
                        await self.manage_quotes(tk, None, round(ts.mid+off,2), 0, fsz, over)
                    elif ts.pos < 0:
                        await self.manage_quotes(tk, round(ts.mid-off,2), None, fsz, 0, over)
                    else:
                        await self.manage_quotes(tk, None, None, 0, 0, over)
                else:
                    # CORE: compute fair value from FILTERED mid
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
        print(f"Final: gross={st.gross():,} net={st.net():+,} reqs={self.api.reqs}")
        for t in TICKERS:
            ts = st.tk[t]
            print(f"  {t}: pos={ts.pos:+,} posts={ts.posts} cancels={ts.cancels}")
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