"""
lob_imbalance_tool.py

Reference implementation of a trader assistance tool that predicts short-horizon direction
from top-5 L2 liquidity imbalance, comparing last 5 book ticks vs prior 5.

Dependencies:
  - Python 3.11+
  - numpy, pandas
  - pyarrow (recommended, for Parquet output)

Install:
  pip install numpy pandas pyarrow

Notes on "tick" definition:
  - This script defines a "book tick" as the end of an atomic update batch (event_end=True),
    similar in spirit to exchange feeds that define atomic transitions for multi-level book updates.
  - Tick duration is open-ended and configured in the SyntheticMarket generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Deque, Any
from collections import deque
import logging
import math
import os

import numpy as np
import pandas as pd


# ----------------------------
# Configuration and utilities
# ----------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def level_weights(levels: int, decay: float = 0.5, normalize: bool = False) -> np.ndarray:
    """Generate nonnegative weights for L book levels."""
    w = (decay ** np.arange(levels)).astype(float)
    if normalize:
        s = w.sum()
        if s > 0:
            w = w / s
    return w


def weighted_depth(bid_sizes: np.ndarray, ask_sizes: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Compute weighted bid/ask depth given sizes and weights."""
    Db = float(np.dot(bid_sizes, w))
    Da = float(np.dot(ask_sizes, w))
    return Db, Da


def imbalance(Db: float, Da: float, eps: float = 1e-12) -> float:
    """Normalized depth imbalance."""
    return (Db - Da) / (Db + Da + eps)


def last5_vs_prior5(series10: List[float]) -> Tuple[float, float, float]:
    """
    series10 is length-10 in chronological order (oldest -> newest):
      prior = first 5, recent = last 5.
    """
    if len(series10) != 10:
        raise ValueError("Need exactly 10 values for last5-vs-prior5.")
    prior = series10[:5]
    recent = series10[5:]
    mean_prior = float(np.mean(prior))
    mean_recent = float(np.mean(recent))
    delta = mean_recent - mean_prior
    return mean_recent, mean_prior, delta


def welch_t_stat(recent: List[float], prior: List[float], eps: float = 1e-12) -> float:
    """Welch-style t-statistic for difference in means between two samples."""
    n1, n0 = len(recent), len(prior)
    if n1 < 2 or n0 < 2:
        return 0.0
    m1, m0 = float(np.mean(recent)), float(np.mean(prior))
    v1, v0 = float(np.var(recent, ddof=1)), float(np.var(prior, ddof=1))
    denom = math.sqrt(v1 / n1 + v0 / n0 + eps)
    return (m1 - m0) / denom


# ----------------------------
# Rolling stats and buffers
# ----------------------------

@dataclass
class RollingWindowStats:
    """Rolling mean/std over a fixed window, maintained with O(1) updates."""
    window: int
    dq: Deque[float] = field(default_factory=deque)
    sum_: float = 0.0
    sumsq: float = 0.0

    def push(self, x: float) -> None:
        if len(self.dq) == self.window:
            old = self.dq.popleft()
            self.sum_ -= old
            self.sumsq -= old * old
        self.dq.append(x)
        self.sum_ += x
        self.sumsq += x * x

    def mean(self) -> Optional[float]:
        n = len(self.dq)
        return (self.sum_ / n) if n else None

    def std(self, ddof: int = 1) -> Optional[float]:
        n = len(self.dq)
        if n <= ddof:
            return None
        mu = self.sum_ / n
        pop_var = (self.sumsq / n) - mu * mu
        pop_var = max(pop_var, 0.0)
        # convert population variance to sample variance
        sample_var = pop_var * (n / (n - ddof))
        return math.sqrt(sample_var)


# ----------------------------
# Market data message types
# ----------------------------

@dataclass(frozen=True)
class PriceLevelUpdate:
    """
    Aggregated price-level update (market-by-price).
    This approximates real L2 aggregation feeds where each update sets the displayed size at a price.
    """
    symbol: str
    side: str                # 'B' bid, 'A' ask
    price: float
    size: int                # if 0 => remove level
    event_ts_ns: int         # event timestamp in ns (open-ended)
    recv_ts_ns: int          # local receive timestamp
    seq: int                 # message sequence number (per symbol in this synthetic feed)
    event_end: bool          # ends atomic update batch (safe to emit book tick)


@dataclass(frozen=True)
class L2Snapshot:
    """Canonical top-L snapshot emitted at the end of an atomic update batch."""
    symbol: str
    tick: int
    event_ts_ns: int
    recv_ts_ns: int
    seq: int
    bid_prices: Tuple[float, ...]
    bid_sizes: Tuple[int, ...]
    ask_prices: Tuple[float, ...]
    ask_sizes: Tuple[int, ...]

    @property
    def mid(self) -> Optional[float]:
        if not self.bid_prices or not self.ask_prices:
            return None
        return (self.bid_prices[0] + self.ask_prices[0]) / 2.0

    @property
    def spread(self) -> Optional[float]:
        if not self.bid_prices or not self.ask_prices:
            return None
        return self.ask_prices[0] - self.bid_prices[0]


# ----------------------------
# Sequencing reorder buffer
# ----------------------------

@dataclass
class SeqReorderBuffer:
    """
    Small per-symbol reorder buffer for message sequence numbers.

    If packets/messages arrive modestly out of order, this buffer reorders them and emits in-order updates.
    If the buffer overflows, it flags a gap and you should resync (e.g., request a snapshot refresh).
    """
    max_buffer: int = 200
    next_seq: Dict[str, int] = field(default_factory=dict)
    buffers: Dict[str, Dict[int, PriceLevelUpdate]] = field(default_factory=dict)
    gap_flag: Dict[str, bool] = field(default_factory=dict)

    def push(self, u: PriceLevelUpdate) -> List[PriceLevelUpdate]:
        sym = u.symbol
        if sym not in self.buffers:
            self.buffers[sym] = {}
            self.next_seq[sym] = u.seq
            self.gap_flag[sym] = False

        buf = self.buffers[sym]
        buf[u.seq] = u

        if len(buf) > self.max_buffer:
            self.gap_flag[sym] = True
            out = [buf[k] for k in sorted(buf.keys())]
            buf.clear()
            self.next_seq[sym] = out[-1].seq + 1
            return out

        out: List[PriceLevelUpdate] = []
        while self.next_seq[sym] in buf:
            out.append(buf.pop(self.next_seq[sym]))
            self.next_seq[sym] += 1
        return out


# ----------------------------
# Book state builder (market-by-price)
# ----------------------------

@dataclass
class BookState:
    bids: Dict[float, int] = field(default_factory=dict)  # price -> size
    asks: Dict[float, int] = field(default_factory=dict)

    def apply_update(self, u: PriceLevelUpdate) -> None:
        book = self.bids if u.side == "B" else self.asks
        if u.size <= 0:
            book.pop(u.price, None)
        else:
            book[u.price] = int(u.size)

    def top_n(self, side: str, n: int) -> List[Tuple[float, int]]:
        book = self.bids if side == "B" else self.asks
        items = list(book.items())
        items.sort(key=lambda x: x[0], reverse=(side == "B"))
        return items[:n]


@dataclass
class BookBuilder:
    levels_out: int = 5
    states: Dict[str, BookState] = field(default_factory=dict)
    ticks: Dict[str, int] = field(default_factory=dict)

    def reset_symbol(self, sym: str) -> None:
        self.states.pop(sym, None)
        self.ticks.pop(sym, None)

    def process_update(self, u: PriceLevelUpdate) -> Optional[L2Snapshot]:
        if u.symbol not in self.states:
            self.states[u.symbol] = BookState()
            self.ticks[u.symbol] = 0

        self.states[u.symbol].apply_update(u)

        if not u.event_end:
            return None

        self.ticks[u.symbol] += 1
        bb = self.states[u.symbol].top_n("B", self.levels_out)
        aa = self.states[u.symbol].top_n("A", self.levels_out)

        return L2Snapshot(
            symbol=u.symbol,
            tick=self.ticks[u.symbol],
            event_ts_ns=u.event_ts_ns,
            recv_ts_ns=u.recv_ts_ns,
            seq=u.seq,
            bid_prices=tuple([p for p, _ in bb]),
            bid_sizes=tuple([sz for _, sz in bb]),
            ask_prices=tuple([p for p, _ in aa]),
            ask_sizes=tuple([sz for _, sz in aa]),
        )


# ----------------------------
# Feature + signal engine
# ----------------------------

@dataclass
class SymbolEngineConfig:
    levels: int = 5
    weight_decay: float = 0.5
    weight_normalize: bool = False
    eps: float = 1e-12

    # windowing
    warmup_ticks: int = 10          # need 10 ticks
    long_window: int = 500          # rolling z-score window for ΔI

    # gates/thresholds
    z_threshold: float = 1.0
    t_threshold: float = 2.0
    stale_latency_ms: float = 50.0
    spread_gate_max: Optional[float] = None

    # confidence shaping
    confidence_kappa: float = 1.25


@dataclass(frozen=True)
class SignalRecord:
    symbol: str
    tick: int
    event_ts_ns: int
    recv_ts_ns: int
    latency_ms: float
    mid: Optional[float]
    spread: Optional[float]

    Db: float
    Da: float
    I: float

    mean_recent_I: Optional[float]
    mean_prior_I: Optional[float]
    delta_I: Optional[float]
    z_delta_I: Optional[float]
    t_stat: Optional[float]

    signal: str
    confidence: float
    reason: str


@dataclass
class SymbolEngine:
    symbol: str
    cfg: SymbolEngineConfig
    w: np.ndarray = field(init=False)
    ring_I: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    delta_stats: RollingWindowStats = field(init=False)

    def __post_init__(self) -> None:
        self.w = level_weights(self.cfg.levels, decay=self.cfg.weight_decay, normalize=self.cfg.weight_normalize)
        self.delta_stats = RollingWindowStats(window=self.cfg.long_window)

    def _fresh(self, event_ts_ns: int, recv_ts_ns: int) -> Tuple[bool, float]:
        latency_ms = (recv_ts_ns - event_ts_ns) / 1e6
        return (latency_ms <= self.cfg.stale_latency_ms), latency_ms

    def process_snapshot(self, snap: L2Snapshot) -> SignalRecord:
        bid_sz = np.array(list(snap.bid_sizes) + [0] * (self.cfg.levels - len(snap.bid_sizes)), dtype=float)[: self.cfg.levels]
        ask_sz = np.array(list(snap.ask_sizes) + [0] * (self.cfg.levels - len(snap.ask_sizes)), dtype=float)[: self.cfg.levels]

        Db, Da = weighted_depth(bid_sz, ask_sz, self.w)
        I = imbalance(Db, Da, eps=self.cfg.eps)
        self.ring_I.append(I)

        fresh, latency_ms = self._fresh(snap.event_ts_ns, snap.recv_ts_ns)
        mid, spread = snap.mid, snap.spread

        if len(self.ring_I) < self.cfg.warmup_ticks:
            return SignalRecord(
                symbol=self.symbol, tick=snap.tick, event_ts_ns=snap.event_ts_ns, recv_ts_ns=snap.recv_ts_ns,
                latency_ms=latency_ms, mid=mid, spread=spread,
                Db=Db, Da=Da, I=I,
                mean_recent_I=None, mean_prior_I=None, delta_I=None, z_delta_I=None, t_stat=None,
                signal="WARMING", confidence=0.0, reason="insufficient_history",
            )

        series10 = list(self.ring_I)
        prior = series10[:5]
        recent = series10[5:]
        mean_recent, mean_prior, delta_I = last5_vs_prior5(series10)
        t_stat = welch_t_stat(recent=recent, prior=prior, eps=self.cfg.eps)

        self.delta_stats.push(delta_I)
        mu = self.delta_stats.mean()
        sd = self.delta_stats.std(ddof=1)
        z = (delta_I - mu) / (sd + self.cfg.eps) if (mu is not None and sd is not None and sd > 0) else 0.0

        if not fresh:
            return SignalRecord(
                symbol=self.symbol, tick=snap.tick, event_ts_ns=snap.event_ts_ns, recv_ts_ns=snap.recv_ts_ns,
                latency_ms=latency_ms, mid=mid, spread=spread,
                Db=Db, Da=Da, I=I,
                mean_recent_I=mean_recent, mean_prior_I=mean_prior, delta_I=delta_I, z_delta_I=z, t_stat=t_stat,
                signal="STALE", confidence=0.0, reason="stale_latency",
            )

        if self.cfg.spread_gate_max is not None and spread is not None and spread > self.cfg.spread_gate_max:
            return SignalRecord(
                symbol=self.symbol, tick=snap.tick, event_ts_ns=snap.event_ts_ns, recv_ts_ns=snap.recv_ts_ns,
                latency_ms=latency_ms, mid=mid, spread=spread,
                Db=Db, Da=Da, I=I,
                mean_recent_I=mean_recent, mean_prior_I=mean_prior, delta_I=delta_I, z_delta_I=z, t_stat=t_stat,
                signal="NEUTRAL", confidence=0.0, reason="spread_gate",
            )

        signal = "NEUTRAL"
        reason = "below_thresholds"
        if abs(z) >= self.cfg.z_threshold and abs(t_stat) >= self.cfg.t_threshold:
            if delta_I > 0:
                signal, reason = "UP", "delta_positive_thresholded"
            elif delta_I < 0:
                signal, reason = "DOWN", "delta_negative_thresholded"

        conf = logistic(self.cfg.confidence_kappa * abs(z)) * logistic(0.75 * abs(t_stat))
        if signal == "NEUTRAL":
            conf *= 0.5

        return SignalRecord(
            symbol=self.symbol, tick=snap.tick, event_ts_ns=snap.event_ts_ns, recv_ts_ns=snap.recv_ts_ns,
            latency_ms=latency_ms, mid=mid, spread=spread,
            Db=Db, Da=Da, I=I,
            mean_recent_I=mean_recent, mean_prior_I=mean_prior, delta_I=delta_I, z_delta_I=z, t_stat=t_stat,
            signal=signal, confidence=float(conf), reason=reason,
        )


# ----------------------------
# Synthetic market generator
# ----------------------------

@dataclass
class SymbolSimConfig:
    tick_size: float = 0.01
    base_mid: float = 100.0
    base_spread: float = 0.02
    base_depth: int = 800
    depth_decay: float = 0.85

    imbalance_trend: float = 0.0     # >0 strengthens bids vs asks over time
    mid_drift_per_tick: float = 0.0  # shifts price levels to create directional mid drift

    noise_scale: float = 0.05
    event_updates_per_tick: Tuple[int, int] = (1, 3)

    latency_ms_mean: float = 5.0
    latency_ms_std: float = 2.0

    gap_probability: float = 0.0     # simulate message seq gaps (for resync logic)


class SyntheticMarket:
    """
    Produces price-level updates (market-by-price) grouped into atomic events.
    This is not a market simulator in the execution sense; it only generates plausible L2 dynamics for testing.
    """

    def __init__(self, symbols: Dict[str, SymbolSimConfig], levels: int = 10, seed: int = 1):
        self.cfgs = symbols
        self.levels = levels
        self.rng = np.random.default_rng(seed)
        self.books: Dict[str, BookState] = {s: BookState() for s in symbols}
        self.seq: Dict[str, int] = {s: 0 for s in symbols}
        self.event_ts_ns: Dict[str, int] = {s: int(1e9) for s in symbols}  # start time
        self.tick_counter: Dict[str, int] = {s: 0 for s in symbols}

        for s in symbols:
            self._init_book(s)

    def _init_book(self, s: str) -> None:
        cfg = self.cfgs[s]
        mid = cfg.base_mid
        spread = cfg.base_spread
        best_bid = round(mid - spread / 2, 2)
        best_ask = round(mid + spread / 2, 2)

        for k in range(1, self.levels + 1):
            bid_px = round(best_bid - (k - 1) * cfg.tick_size, 2)
            ask_px = round(best_ask + (k - 1) * cfg.tick_size, 2)
            bid_sz = int(cfg.base_depth * (cfg.depth_decay ** (k - 1)))
            ask_sz = int(cfg.base_depth * (cfg.depth_decay ** (k - 1)))
            self.books[s].bids[bid_px] = bid_sz
            self.books[s].asks[ask_px] = ask_sz

    def emit_full_book_refresh(self, s: str) -> List[PriceLevelUpdate]:
        """Emit a full refresh of current book state (needed to seed a book builder)."""
        cfg = self.cfgs[s]
        self.event_ts_ns[s] += int(1e6)

        items: List[Tuple[str, float, int]] = []
        for px, sz in self.books[s].bids.items():
            items.append(("B", float(px), int(sz)))
        for px, sz in self.books[s].asks.items():
            items.append(("A", float(px), int(sz)))

        # deterministic ordering
        items.sort(key=lambda x: (x[0], x[1] if x[0] == "A" else -x[1]))

        out: List[PriceLevelUpdate] = []
        for i, (side, px, sz) in enumerate(items):
            self.seq[s] += 1
            seq = self.seq[s]
            delay_ms = max(0.0, float(self.rng.normal(cfg.latency_ms_mean, cfg.latency_ms_std)))
            recv_ts_ns = self.event_ts_ns[s] + int(delay_ms * 1e6)
            out.append(PriceLevelUpdate(
                symbol=s, side=side, price=px, size=sz,
                event_ts_ns=self.event_ts_ns[s], recv_ts_ns=recv_ts_ns, seq=seq,
                event_end=(i == len(items) - 1),
            ))
        return out

    def _level_probs(self) -> np.ndarray:
        raw = np.array([0.6 ** (k - 1) for k in range(1, self.levels + 1)], dtype=float)
        raw /= raw.sum()
        return raw

    def _price_at_level(self, s: str, side: str, level: int) -> float:
        top = self.books[s].top_n(side, self.levels)
        if len(top) >= level:
            return top[level - 1][0]
        # fall back to extrapolation
        cfg = self.cfgs[s]
        best = (cfg.base_mid - cfg.base_spread / 2) if side == "B" else (cfg.base_mid + cfg.base_spread / 2)
        return round(best + (level - 1) * cfg.tick_size * (1 if side == "A" else -1), 2)

    def _shift_prices(self, s: str, drift: float) -> None:
        """Shift entire top-N ladder to move mid price in tick increments."""
        cfg = self.cfgs[s]
        bb = self.books[s].top_n("B", 1)
        aa = self.books[s].top_n("A", 1)
        if not bb or not aa:
            return
        mid = (bb[0][0] + aa[0][0]) / 2.0 + drift

        best_bid = round(mid - cfg.base_spread / 2, 2)
        best_ask = round(mid + cfg.base_spread / 2, 2)

        top_bid = self.books[s].top_n("B", self.levels)
        top_ask = self.books[s].top_n("A", self.levels)
        bid_sizes = [sz for _, sz in top_bid]
        ask_sizes = [sz for _, sz in top_ask]
        bid_sizes += [cfg.base_depth] * (self.levels - len(bid_sizes))
        ask_sizes += [cfg.base_depth] * (self.levels - len(ask_sizes))

        new_bids, new_asks = {}, {}
        for k in range(1, self.levels + 1):
            new_bids[round(best_bid - (k - 1) * cfg.tick_size, 2)] = int(bid_sizes[k - 1])
            new_asks[round(best_ask + (k - 1) * cfg.tick_size, 2)] = int(ask_sizes[k - 1])
        self.books[s].bids = new_bids
        self.books[s].asks = new_asks

    def generate_events(self, s: str, n_ticks: int) -> List[PriceLevelUpdate]:
        """Generate updates for n atomic ticks for a single symbol."""
        cfg = self.cfgs[s]
        out: List[PriceLevelUpdate] = []

        for _ in range(n_ticks):
            self.tick_counter[s] += 1
            # tick duration is open-ended; here we use 5ms as a placeholder
            self.event_ts_ns[s] += int(5e6)

            if cfg.mid_drift_per_tick != 0.0:
                self._shift_prices(s, cfg.mid_drift_per_tick)

            m = int(self.rng.integers(cfg.event_updates_per_tick[0], cfg.event_updates_per_tick[1] + 1))
            trend = cfg.imbalance_trend

            for j in range(m):
                side = "B" if (self.rng.random() < 0.5) else "A"
                if trend > 0 and self.rng.random() < 0.65:
                    side = "B"
                if trend < 0 and self.rng.random() < 0.65:
                    side = "A"

                level = int(self.rng.choice(np.arange(1, self.levels + 1), p=self._level_probs()))
                px = self._price_at_level(s, side, level)
                book = self.books[s].bids if side == "B" else self.books[s].asks
                old_sz = int(book.get(px, 0))

                noise = max(0.0, 1.0 + float(self.rng.normal(0, cfg.noise_scale)))
                if side == "B":
                    new_sz = int(max(0, old_sz * noise + (trend * 50)))
                else:
                    new_sz = int(max(0, old_sz * noise - (trend * 50)))

                if self.rng.random() < 0.02:
                    new_sz = 0

                self.seq[s] += 1
                seq = self.seq[s]

                # optionally simulate a missing sequence number
                if cfg.gap_probability > 0 and self.rng.random() < cfg.gap_probability:
                    self.seq[s] += 1
                    seq = self.seq[s]

                delay_ms = max(0.0, float(self.rng.normal(cfg.latency_ms_mean, cfg.latency_ms_std)))
                recv_ts_ns = self.event_ts_ns[s] + int(delay_ms * 1e6)

                u = PriceLevelUpdate(
                    symbol=s, side=side, price=px, size=new_sz,
                    event_ts_ns=self.event_ts_ns[s], recv_ts_ns=recv_ts_ns,
                    seq=seq, event_end=(j == m - 1),
                )
                # apply immediately to the generator's internal book
                self.books[s].apply_update(u)
                out.append(u)

        return out


# ----------------------------
# Backtest / labeling / metrics
# ----------------------------

@dataclass
class RoundConfig:
    name: str
    active_symbols: List[str]
    ticks_per_symbol: int
    imbalance_trend: Dict[str, float] = field(default_factory=dict)       # per-round overrides
    mid_drift_per_tick: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    levels_out: int = 5
    horizon_h: int = 3
    deadband: float = 0.0
    parquet_dir: str = "lob_out"
    write_parquet: bool = True


def label_direction(mid: List[Optional[float]], h: int, deadband: float = 0.0) -> List[Optional[int]]:
    """+1 up, -1 down, 0 flat, None if insufficient future data."""
    n = len(mid)
    out: List[Optional[int]] = [None] * n
    for i in range(n):
        if i + h >= n or mid[i] is None or mid[i + h] is None:
            out[i] = None
            continue
        diff = mid[i + h] - mid[i]
        if diff > deadband:
            out[i] = 1
        elif diff < -deadband:
            out[i] = -1
        else:
            out[i] = 0
    return out


def signal_to_label(sig: str) -> Optional[int]:
    return {"UP": 1, "DOWN": -1, "NEUTRAL": 0}.get(sig)


def classification_report(y_true: List[int], y_pred: List[int], labels: List[int] = [-1, 0, 1]) -> Dict[str, Any]:
    """Precision/recall/F1 per class + macro averages + accuracy."""
    rep: Dict[str, Any] = {}
    for lab in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp == lab)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lab and yp == lab)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        rep[str(lab)] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    rep["macro_avg"] = {
        "precision": float(np.mean([rep[str(l)]["precision"] for l in labels])),
        "recall": float(np.mean([rep[str(l)]["recall"] for l in labels])),
        "f1": float(np.mean([rep[str(l)]["f1"] for l in labels])),
        "support": len(y_true),
    }
    rep["accuracy"] = (sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)) if y_true else 0.0
    return rep


def sharpe_like(returns: List[float], eps: float = 1e-12) -> float:
    """Mean/Std without risk-free rate; illustrative only."""
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    return mu / (sd + eps)


def strategy_returns(mid: List[Optional[float]], signals: List[str], h: int) -> List[float]:
    """Naive position = signal label; P&L = position * (mid[t+h]-mid[t])."""
    out: List[float] = []
    for t in range(len(mid)):
        if t + h >= len(mid) or mid[t] is None or mid[t + h] is None:
            continue
        pos = signal_to_label(signals[t])
        if pos is None:
            continue
        out.append(float(pos) * (mid[t + h] - mid[t]))
    return out


def run_multi_round_backtest(
    sim_cfgs: Dict[str, SymbolSimConfig],
    rounds: List[RoundConfig],
    engine_cfg: SymbolEngineConfig,
    bt_cfg: BacktestConfig,
    seed: int = 1,
    reorder_buffer_size: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingestion pipeline:
      recv-order events -> reorder buffer -> book builder -> top-5 snapshots -> signal engine
    """
    market = SyntheticMarket(symbols=sim_cfgs, levels=max(10, bt_cfg.levels_out), seed=seed)
    reorder = SeqReorderBuffer(max_buffer=reorder_buffer_size)
    builder = BookBuilder(levels_out=bt_cfg.levels_out)

    engines: Dict[str, SymbolEngine] = {}
    seeded_symbols: set[str] = set()

    records: List[SignalRecord] = []
    snapshots: List[Dict[str, Any]] = []

    def ensure_engine(sym: str) -> SymbolEngine:
        if sym not in engines:
            engines[sym] = SymbolEngine(symbol=sym, cfg=engine_cfg)
        return engines[sym]

    def record_snapshot(snap: L2Snapshot, rnd_name: str) -> None:
        rec = ensure_engine(snap.symbol).process_snapshot(snap)
        records.append(rec)
        snapshots.append({
            "round": rnd_name,
            "symbol": snap.symbol,
            "tick": snap.tick,
            "event_ts_ns": snap.event_ts_ns,
            "recv_ts_ns": snap.recv_ts_ns,
            "seq": snap.seq,
            "mid": snap.mid,
            "spread": snap.spread,
            **{f"bid_px_{i+1}": (snap.bid_prices[i] if i < len(snap.bid_prices) else np.nan) for i in range(bt_cfg.levels_out)},
            **{f"bid_sz_{i+1}": (snap.bid_sizes[i] if i < len(snap.bid_sizes) else 0) for i in range(bt_cfg.levels_out)},
            **{f"ask_px_{i+1}": (snap.ask_prices[i] if i < len(snap.ask_prices) else np.nan) for i in range(bt_cfg.levels_out)},
            **{f"ask_sz_{i+1}": (snap.ask_sizes[i] if i < len(snap.ask_sizes) else 0) for i in range(bt_cfg.levels_out)},
        })

    for rnd in rounds:
        logging.info("Starting %s with symbols=%s", rnd.name, rnd.active_symbols)

        # Apply per-round overrides
        for sym, tr in rnd.imbalance_trend.items():
            market.cfgs[sym].imbalance_trend = tr
        for sym, dr in rnd.mid_drift_per_tick.items():
            market.cfgs[sym].mid_drift_per_tick = dr

        # Generate events for active symbols
        events: List[PriceLevelUpdate] = []
        for sym in rnd.active_symbols:
            if sym not in seeded_symbols:
                events.extend(market.emit_full_book_refresh(sym))
                seeded_symbols.add(sym)
            events.extend(market.generate_events(sym, n_ticks=rnd.ticks_per_symbol))

        # Ingestion order = receive timestamp (interleaves symbols)
        events.sort(key=lambda e: (e.recv_ts_ns, e.symbol, e.seq))

        for u in events:
            # per-symbol reorder
            for uu in reorder.push(u):
                if reorder.gap_flag.get(uu.symbol, False):
                    # force resync: clear builder and engine state; next appearance will re-seed via refresh
                    logging.warning("Seq gap/overflow detected for %s: forcing resync.", uu.symbol)
                    builder.reset_symbol(uu.symbol)
                    engines.pop(uu.symbol, None)
                    seeded_symbols.discard(uu.symbol)
                    continue

                snap = builder.process_update(uu)
                if snap is not None:
                    record_snapshot(snap, rnd.name)

    df_signals = pd.DataFrame([r.__dict__ for r in records])
    df_snaps = pd.DataFrame(snapshots)

    # Merge mid into signal records for labeling
    df = df_signals.merge(df_snaps[["symbol", "tick", "round", "mid"]], on=["symbol", "tick"], how="left", suffixes=("", "_snap"))
    if "mid_snap" in df.columns:
        df["mid"] = df["mid"].where(df["mid"].notna(), df["mid_snap"])
        df = df.drop(columns=["mid_snap"])

    # Label and evaluate
    metrics: Dict[str, Any] = {"per_symbol": {}, "aggregate": {}}
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    returns_all: List[float] = []

    for sym in sorted(df["symbol"].unique()):
        sdf = df[df["symbol"] == sym].sort_values("tick").reset_index(drop=True)
        y_true = label_direction(list(sdf["mid"]), h=bt_cfg.horizon_h, deadband=bt_cfg.deadband)
        y_pred = [signal_to_label(s) for s in sdf["signal"].tolist()]

        sdf["y_true"] = y_true
        sdf["y_pred"] = y_pred

        mask = sdf["y_true"].notna() & sdf["y_pred"].notna() & sdf["signal"].isin(["UP", "DOWN", "NEUTRAL"])
        yt = [int(x) for x in sdf.loc[mask, "y_true"].tolist()]
        yp = [int(x) for x in sdf.loc[mask, "y_pred"].tolist()]

        rep = classification_report(yt, yp) if yt else {}
        rets = strategy_returns(list(sdf["mid"]), sdf["signal"].tolist(), h=bt_cfg.horizon_h)
        sr = sharpe_like(rets)

        metrics["per_symbol"][sym] = {
            "classification": rep,
            "sharpe_like": sr,
            "n_eval": len(yt),
            "n_returns": len(rets),
        }

        y_true_all.extend(yt)
        y_pred_all.extend(yp)
        returns_all.extend(rets)

    metrics["aggregate"]["classification"] = classification_report(y_true_all, y_pred_all) if y_true_all else {}
    metrics["aggregate"]["sharpe_like"] = sharpe_like(returns_all)
    metrics["aggregate"]["n_eval"] = len(y_true_all)
    metrics["aggregate"]["n_returns"] = len(returns_all)

    # Store Parquet for replay
    if bt_cfg.write_parquet:
        os.makedirs(bt_cfg.parquet_dir, exist_ok=True)
        snap_path = os.path.join(bt_cfg.parquet_dir, "snapshots.parquet")
        sig_path = os.path.join(bt_cfg.parquet_dir, "signals.parquet")
        try:
            df_snaps.to_parquet(snap_path, index=False)
            df.to_parquet(sig_path, index=False)
            logging.info("Wrote %s and %s", snap_path, sig_path)
        except Exception as e:
            logging.exception("Failed to write Parquet (install pyarrow?): %s", e)
            metrics["storage_error"] = str(e)

    return df, metrics


# ----------------------------
# Default multi-round demo (2 -> 5 symbols)
# ----------------------------

def default_sim_configs() -> Dict[str, SymbolSimConfig]:
    return {
        "AAA": SymbolSimConfig(base_mid=100.0, imbalance_trend=0.8, mid_drift_per_tick=0.01),
        "BBB": SymbolSimConfig(base_mid=50.0,  imbalance_trend=-0.6, mid_drift_per_tick=-0.01),
        "CCC": SymbolSimConfig(base_mid=80.0,  imbalance_trend=0.0, mid_drift_per_tick=0.0),
        "DDD": SymbolSimConfig(base_mid=120.0, imbalance_trend=0.4, mid_drift_per_tick=0.0),
        "EEE": SymbolSimConfig(base_mid=30.0,  imbalance_trend=-0.4, mid_drift_per_tick=0.0),
    }


def default_rounds() -> List[RoundConfig]:
    return [
        RoundConfig(name="round_1", active_symbols=["AAA", "BBB"], ticks_per_symbol=200),
        RoundConfig(name="round_2", active_symbols=["AAA", "BBB", "CCC"], ticks_per_symbol=200,
                    imbalance_trend={"CCC": 0.7}, mid_drift_per_tick={"CCC": 0.01}),
        RoundConfig(name="round_3", active_symbols=["AAA", "BBB", "CCC", "DDD"], ticks_per_symbol=200,
                    imbalance_trend={"DDD": -0.5}, mid_drift_per_tick={"DDD": -0.01}),
        RoundConfig(name="round_4", active_symbols=["AAA", "BBB", "CCC", "DDD", "EEE"], ticks_per_symbol=200,
                    imbalance_trend={"EEE": 0.6}, mid_drift_per_tick={"EEE": 0.01}),
    ]


def main() -> None:
    setup_logging("INFO")

    sim_cfgs = default_sim_configs()
    rounds = default_rounds()

    engine_cfg = SymbolEngineConfig(
        levels=5,
        weight_decay=0.5,
        long_window=300,
        z_threshold=0.8,
        t_threshold=1.8,
        stale_latency_ms=30.0,
        spread_gate_max=0.10,
        confidence_kappa=1.25,
    )

    bt_cfg = BacktestConfig(
        levels_out=5,
        horizon_h=3,
        deadband=0.0,
        parquet_dir="lob_out",
        write_parquet=True,
    )

    df, metrics = run_multi_round_backtest(
        sim_cfgs=sim_cfgs,
        rounds=rounds,
        engine_cfg=engine_cfg,
        bt_cfg=bt_cfg,
        seed=42,
        reorder_buffer_size=200,
    )

    print("\n=== Aggregate metrics ===")
    print("n_eval:", metrics["aggregate"]["n_eval"])
    print("Sharpe-like:", round(metrics["aggregate"]["sharpe_like"], 4))
    print("Classification (macro F1):", round(metrics["aggregate"]["classification"].get("macro_avg", {}).get("f1", 0.0), 4))
    print("Accuracy:", round(metrics["aggregate"]["classification"].get("accuracy", 0.0), 4))

    # Example output table: last few rows for a symbol
    for sym in ["AAA", "BBB"]:
        sdf = df[df["symbol"] == sym].sort_values("tick").tail(10)
        cols = ["symbol", "tick", "mid", "spread", "I", "delta_I", "z_delta_I", "t_stat", "signal", "confidence", "reason"]
        print(f"\n=== Sample tail for {sym} ===")
        print(sdf[cols].to_string(index=False))


if __name__ == "__main__":
    main()
