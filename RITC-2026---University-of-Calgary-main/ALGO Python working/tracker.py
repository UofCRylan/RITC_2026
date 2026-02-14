"""
Performance Tracker for RIT Algo Market Maker
==============================================
Records tick-by-tick snapshots of P&L, positions, market data, and bot
diagnostics.  Writes to CSV for post-heat analysis and prints a live
summary to the console.

Usage (standalone – just observes, doesn't trade):
    python tracker.py

Usage (integrated – import into a.py):
    from tracker import PerformanceTracker
    tracker = PerformanceTracker(api, tickers)
    # inside your loop:
    tracker.record(tick, period, state, params, news, vol, momentum, corr)
    # at the end:
    tracker.save()
    tracker.print_summary()
"""

from __future__ import annotations
import csv
import os
import time
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple
from collections import deque

# ---------------------------------------------------------------------------
#  SNAPSHOT DATA
# ---------------------------------------------------------------------------

@dataclass
class TickSnapshot:
    """One row of data captured at a single tick."""
    # Timing
    timestamp: float           # time.time()
    tick: int
    period: int
    ticks_until_close: int

    # Account-level
    nlv: float                 # net liquid value (total P&L incl penalties)
    nlv_delta: float           # change since last snapshot
    aggregate_position: int    # sum of |positions|
    aggregate_limit: int
    gross_position: int
    gross_limit: int
    net_position: int          # signed sum
    net_limit: int

    # Per-ticker (stored as dicts keyed by ticker)
    positions: Dict[str, int]
    mids: Dict[str, float]
    spreads: Dict[str, float]
    vwaps: Dict[str, float]
    unrealized: Dict[str, float]
    realized: Dict[str, float]
    momentums: Dict[str, float]
    volatilities: Dict[str, float]
    fill_rates: Dict[str, float]
    news_adjs: Dict[str, float]
    bid_prices: Dict[str, float]
    ask_prices: Dict[str, float]
    bid_sizes: Dict[str, int]
    ask_sizes: Dict[str, int]
    num_fills: Dict[str, int]
    num_posts: Dict[str, int]
    num_cancels: Dict[str, int]


# ---------------------------------------------------------------------------
#  PERFORMANCE TRACKER
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """
    Records tick-by-tick snapshots and provides P&L analytics.
    """

    def __init__(
        self,
        api,             # RITClient instance
        tickers: List[str],
        output_dir: str = ".",
        snapshot_interval: float = 1.0,  # seconds between snapshots
    ):
        self.api = api
        self.tickers = tickers
        self.output_dir = output_dir
        self.snapshot_interval = snapshot_interval

        # Snapshot storage
        self.snapshots: List[TickSnapshot] = []

        # Running state
        self.last_snapshot_ts: float = 0.0
        self.starting_nlv: Optional[float] = None
        self.last_nlv: float = 0.0
        self.peak_nlv: float = 0.0
        self.trough_nlv: float = 0.0
        self.max_drawdown: float = 0.0

        # Per-day P&L tracking (market close = every 60 ticks)
        self.day_start_nlv: float = 0.0
        self.day_pnls: List[float] = []
        self.last_day_num: int = -1

        # Heat-level stats
        self.heat_start_ts: float = time.time()
        self.total_fills: int = 0
        self.total_posts: int = 0
        self.total_cancels: int = 0
        self.penalty_events: int = 0  # times we were over aggregate limit at close

        # CSV setup
        import pandas as pd
        timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(output_dir, f"perf_{timestamp_str}.csv")
        self.csv_file = None
        self.csv_writer = None

    def _init_csv(self, snap: TickSnapshot) -> None:
        """Create CSV with headers on first snapshot."""
        self.csv_file = open(self.csv_path, "w", newline="")
        headers = [
            "timestamp", "tick", "period", "ticks_until_close",
            "nlv", "nlv_delta", "pnl_from_start",
            "aggregate_position", "aggregate_limit",
            "gross_position", "gross_limit",
            "net_position", "net_limit",
        ]
        for t in self.tickers:
            headers.extend([
                f"{t}_pos", f"{t}_mid", f"{t}_spread", f"{t}_vwap",
                f"{t}_unrealized", f"{t}_realized",
                f"{t}_momentum", f"{t}_vol", f"{t}_fill_rate",
                f"{t}_news_adj",
                f"{t}_bid_px", f"{t}_ask_px",
                f"{t}_bid_sz", f"{t}_ask_sz",
                f"{t}_fills", f"{t}_posts", f"{t}_cancels",
            ])
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(headers)

    def _write_csv_row(self, snap: TickSnapshot) -> None:
        """Write one snapshot row to CSV."""
        if self.csv_writer is None:
            self._init_csv(snap)

        pnl_from_start = snap.nlv - (self.starting_nlv or snap.nlv)

        row = [
            f"{snap.timestamp:.3f}", snap.tick, snap.period, snap.ticks_until_close,
            f"{snap.nlv:.2f}", f"{snap.nlv_delta:.2f}", f"{pnl_from_start:.2f}",
            snap.aggregate_position, snap.aggregate_limit,
            snap.gross_position, snap.gross_limit,
            snap.net_position, snap.net_limit,
        ]
        for t in self.tickers:
            row.extend([
                snap.positions.get(t, 0),
                f"{snap.mids.get(t, 0):.4f}",
                f"{snap.spreads.get(t, 0):.4f}",
                f"{snap.vwaps.get(t, 0):.4f}",
                f"{snap.unrealized.get(t, 0):.2f}",
                f"{snap.realized.get(t, 0):.2f}",
                f"{snap.momentums.get(t, 0):.6f}",
                f"{snap.volatilities.get(t, 0):.6f}",
                f"{snap.fill_rates.get(t, 0):.3f}",
                f"{snap.news_adjs.get(t, 0):.4f}",
                f"{snap.bid_prices.get(t, 0):.2f}",
                f"{snap.ask_prices.get(t, 0):.2f}",
                snap.bid_sizes.get(t, 0),
                snap.ask_sizes.get(t, 0),
                snap.num_fills.get(t, 0),
                snap.num_posts.get(t, 0),
                snap.num_cancels.get(t, 0),
            ])
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def should_record(self) -> bool:
        """Check if enough time has passed for next snapshot."""
        return (time.time() - self.last_snapshot_ts) >= self.snapshot_interval

    def record(
        self,
        tick: int,
        period: int,
        state,          # BotState
        params,         # QuotingParams
        news_tracker,   # NewsTracker
        vol_tracker,    # VolatilityTracker
        mom_tracker,    # MomentumTracker
        corr_tracker,   # CorrelationTracker
    ) -> Optional[TickSnapshot]:
        """
        Take a snapshot. Queries the API for NLV/P&L data.
        Returns the snapshot, or None if skipped (interval not elapsed).
        """
        if not self.should_record():
            return None

        now = time.time()
        self.last_snapshot_ts = now

        # --- Fetch account-level data ---
        trader = self.api.get_trader()
        nlv = float(trader.get("nlv", 0)) if trader else 0.0

        securities = self.api.get_securities()
        limits = self.api.get_limits()

        # Parse limits
        gross, gross_limit, net_pos_from_limits, net_limit = 0, 25000, 0, 25000
        if limits and len(limits) > 0:
            lim = limits[0]
            gross = int(lim.get("gross", 0))
            gross_limit = int(lim.get("gross_limit", 25000))
            net_limit = int(lim.get("net_limit", 25000))

        # --- Per-ticker data ---
        positions: Dict[str, int] = {}
        mids: Dict[str, float] = {}
        spreads: Dict[str, float] = {}
        vwaps: Dict[str, float] = {}
        unrealized_map: Dict[str, float] = {}
        realized_map: Dict[str, float] = {}
        momentums: Dict[str, float] = {}
        volatilities: Dict[str, float] = {}
        fill_rates: Dict[str, float] = {}
        news_adjs: Dict[str, float] = {}
        bid_prices: Dict[str, float] = {}
        ask_prices: Dict[str, float] = {}
        bid_sizes: Dict[str, int] = {}
        ask_sizes: Dict[str, int] = {}
        num_fills: Dict[str, int] = {}
        num_posts: Dict[str, int] = {}
        num_cancels: Dict[str, int] = {}

        if securities:
            for sec in securities:
                t = sec.get("ticker", "")
                if t not in self.tickers:
                    continue
                positions[t] = int(sec.get("position", 0))
                vwaps[t] = float(sec.get("vwap", 0))
                unrealized_map[t] = float(sec.get("unrealized", 0))
                realized_map[t] = float(sec.get("realized", 0))

        for t in self.tickers:
            ts = state.s(t) if state else None
            if ts:
                mids[t] = ts.last_mid or 0.0
                spreads[t] = ts.last_spread or 0.0
                momentums[t] = ts.momentum
                volatilities[t] = ts.volatility
                fill_rates[t] = ts.fill_rate
                num_fills[t] = ts.num_fills
                num_posts[t] = ts.num_posts
                num_cancels[t] = ts.num_cancels
                bid_prices[t] = ts.working.bid_px or 0.0
                ask_prices[t] = ts.working.ask_px or 0.0
                bid_sizes[t] = ts.working.bid_qty
                ask_sizes[t] = ts.working.ask_qty
            else:
                mids[t] = 0.0
                spreads[t] = 0.0

            if news_tracker:
                news_adjs[t] = news_tracker.get_adjustment(t)

        # Aggregate metrics
        agg_pos = sum(abs(v) for v in positions.values())
        net_pos = sum(positions.values())
        agg_limit = state.aggregate_limit if state else 10000

        ticks_per_day = params.ticks_per_day if params else 60
        ticks_until_close = ticks_per_day - (tick % ticks_per_day) if ticks_per_day > 0 else 999

        # --- Update running stats ---
        if self.starting_nlv is None:
            self.starting_nlv = nlv
            self.peak_nlv = nlv
            self.trough_nlv = nlv
            self.day_start_nlv = nlv

        nlv_delta = nlv - self.last_nlv if self.last_nlv != 0 else 0.0
        self.last_nlv = nlv

        if nlv > self.peak_nlv:
            self.peak_nlv = nlv
        drawdown = self.peak_nlv - nlv
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Day tracking (market close = every 60 ticks)
        day_num = tick // ticks_per_day
        if day_num != self.last_day_num and self.last_day_num >= 0:
            # New day boundary — record P&L for the completed day
            day_pnl = nlv - self.day_start_nlv
            self.day_pnls.append(day_pnl)
            self.day_start_nlv = nlv

            # Check if we were over limit at this close
            if agg_pos > agg_limit:
                self.penalty_events += 1
        self.last_day_num = day_num

        # Cumulative fill/post/cancel totals
        self.total_fills = sum(num_fills.values())
        self.total_posts = sum(num_posts.values())
        self.total_cancels = sum(num_cancels.values())

        # --- Build snapshot ---
        snap = TickSnapshot(
            timestamp=now,
            tick=tick,
            period=period,
            ticks_until_close=ticks_until_close,
            nlv=nlv,
            nlv_delta=nlv_delta,
            aggregate_position=agg_pos,
            aggregate_limit=agg_limit,
            gross_position=gross,
            gross_limit=gross_limit,
            net_position=net_pos,
            net_limit=net_limit,
            positions=positions,
            mids=mids,
            spreads=spreads,
            vwaps=vwaps,
            unrealized=unrealized_map,
            realized=realized_map,
            momentums=momentums,
            volatilities=volatilities,
            fill_rates=fill_rates,
            news_adjs=news_adjs,
            bid_prices=bid_prices,
            ask_prices=ask_prices,
            bid_sizes=bid_sizes,
            ask_sizes=ask_sizes,
            num_fills=num_fills,
            num_posts=num_posts,
            num_cancels=num_cancels,
        )

        self.snapshots.append(snap)
        self._write_csv_row(snap)

        return snap

    def print_live_dashboard(self, snap: TickSnapshot) -> None:
        """Print a compact live P&L dashboard to console."""
        pnl = snap.nlv - (self.starting_nlv or snap.nlv)
        elapsed = time.time() - self.heat_start_ts
        pnl_per_min = pnl / max(elapsed / 60, 0.01)

        total_unreal = sum(snap.unrealized.values())
        total_real = sum(snap.realized.values())

        lines = [
            f"\n{'='*70}",
            f"  P&L DASHBOARD  |  T={snap.tick:>3}  P={snap.period}  "
            f"Close in {snap.ticks_until_close} ticks  "
            f"Elapsed {elapsed:.0f}s",
            f"{'='*70}",
            f"  NLV: ${snap.nlv:>10,.2f}  |  P&L: ${pnl:>+10,.2f}  |  "
            f"P&L/min: ${pnl_per_min:>+8,.2f}",
            f"  Peak: ${self.peak_nlv:>10,.2f}  |  Drawdown: ${self.max_drawdown:>8,.2f}  |  "
            f"Penalties: {self.penalty_events}",
            f"  Realized: ${total_real:>8,.2f}  |  Unrealized: ${total_unreal:>8,.2f}",
            f"  Agg: {snap.aggregate_position}/{snap.aggregate_limit} "
            f"({snap.aggregate_position/max(snap.aggregate_limit,1):.0%})  |  "
            f"Net: {snap.net_position:+d}/{snap.net_limit}  |  "
            f"Gross: {snap.gross_position}/{snap.gross_limit}",
            f"  Fills: {self.total_fills}  Posts: {self.total_posts}  "
            f"Cancels: {self.total_cancels}  "
            f"Fill%: {self.total_fills/max(self.total_posts,1):.0%}",
            f"  {'-'*66}",
        ]
        for t in self.tickers:
            pos = snap.positions.get(t, 0)
            mid = snap.mids.get(t, 0)
            spr = snap.spreads.get(t, 0)
            unr = snap.unrealized.get(t, 0)
            rea = snap.realized.get(t, 0)
            mom = snap.momentums.get(t, 0)
            fr = snap.fill_rates.get(t, 0)
            lines.append(
                f"  {t}: pos={pos:>+6d}  mid={mid:>7.2f}  spr={spr:.3f}  "
                f"unrl=${unr:>+8.2f}  real=${rea:>+8.2f}  "
                f"mom={mom:>+.4f}  fr={fr:.0%}"
            )

        # Day P&L history
        if self.day_pnls:
            day_strs = [f"D{i+1}:${p:+.0f}" for i, p in enumerate(self.day_pnls)]
            lines.append(f"  Days: {' | '.join(day_strs)}")

        lines.append(f"{'='*70}")
        print("\n".join(lines))

    def print_summary(self) -> None:
        """Print end-of-heat summary statistics."""
        if not self.snapshots:
            print("[TRACKER] No snapshots recorded.")
            return

        first = self.snapshots[0]
        last = self.snapshots[-1]
        total_pnl = last.nlv - first.nlv
        elapsed = last.timestamp - first.timestamp

        print(f"\n{'#'*70}")
        print(f"  HEAT SUMMARY")
        print(f"{'#'*70}")
        print(f"  Duration: {elapsed:.1f}s  |  Snapshots: {len(self.snapshots)}")
        print(f"  Starting NLV: ${first.nlv:,.2f}  →  Ending NLV: ${last.nlv:,.2f}")
        print(f"  Total P&L: ${total_pnl:>+,.2f}")
        print(f"  P&L/minute: ${total_pnl / max(elapsed/60, 0.01):>+,.2f}")
        print(f"  Peak NLV: ${self.peak_nlv:,.2f}  |  Max Drawdown: ${self.max_drawdown:,.2f}")
        print(f"  Penalty Events (over agg limit at close): {self.penalty_events}")

        # Per-day breakdown
        if self.day_pnls:
            print(f"\n  Per-Day P&L:")
            for i, p in enumerate(self.day_pnls):
                bar = "█" * max(1, int(abs(p) / max(abs(max(self.day_pnls, key=abs)), 1) * 20))
                sign = "+" if p >= 0 else "-"
                print(f"    Day {i+1}: ${p:>+10,.2f}  {sign}{bar}")

            avg_day = sum(self.day_pnls) / len(self.day_pnls)
            std_day = math.sqrt(sum((p - avg_day) ** 2 for p in self.day_pnls) / max(len(self.day_pnls), 1))
            sharpe = avg_day / std_day if std_day > 0 else 0
            win_days = sum(1 for p in self.day_pnls if p > 0)
            print(f"\n    Avg Day P&L: ${avg_day:>+,.2f}  |  Std: ${std_day:,.2f}")
            print(f"    Sharpe (daily): {sharpe:.2f}  |  Win Rate: {win_days}/{len(self.day_pnls)}")

        # Per-ticker breakdown
        if self.snapshots:
            print(f"\n  Per-Ticker Final State:")
            for t in self.tickers:
                pos = last.positions.get(t, 0)
                unr = last.unrealized.get(t, 0)
                rea = last.realized.get(t, 0)
                fills = last.num_fills.get(t, 0)
                posts = last.num_posts.get(t, 0)
                fr = fills / max(posts, 1)
                print(f"    {t}: pos={pos:>+6d}  realized=${rea:>+8.2f}  "
                      f"unrealized=${unr:>+8.2f}  fills={fills}  posts={posts}  fr={fr:.0%}")

        print(f"\n  Total Fills: {self.total_fills}  |  Total Posts: {self.total_posts}")
        print(f"  Overall Fill Rate: {self.total_fills/max(self.total_posts,1):.1%}")
        print(f"  CSV saved to: {self.csv_path}")
        print(f"{'#'*70}\n")

    def save(self) -> None:
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None


# ---------------------------------------------------------------------------
#  STANDALONE MODE – observe without trading
# ---------------------------------------------------------------------------

def run_standalone():
    """
    Run the tracker standalone to observe a practice session.
    Doesn't trade – just records P&L and market data.
    """
    import signal as sig
    import sys

    # Reuse RITClient from a.py
    sys.path.insert(0, os.path.dirname(__file__))
    from a import RITClient

    api = RITClient()
    tickers = ["SPNG", "SMMR", "ATMN", "WNTR"]

    tracker = PerformanceTracker(api, tickers, snapshot_interval=1.0)

    # Minimal state mock for standalone mode
    class MockTickerState:
        def __init__(self):
            self.last_mid = None
            self.last_spread = None
            self.momentum = 0.0
            self.volatility = 0.0
            self.fill_rate = 0.0
            self.num_fills = 0
            self.num_posts = 0
            self.num_cancels = 0
            self.working = type("W", (), {"bid_px": None, "ask_px": None, "bid_qty": 0, "ask_qty": 0})()

    class MockState:
        def __init__(self, tickers):
            self.tickers = tickers
            self.aggregate_limit = 10000
            self._ts = {t: MockTickerState() for t in tickers}
        def s(self, t):
            return self._ts[t]

    state = MockState(tickers)
    running = True

    def stop_handler(*args):
        nonlocal running
        running = False

    sig.signal(sig.SIGINT, stop_handler)

    print(f"[TRACKER] Standalone mode – observing {tickers}")
    print(f"[TRACKER] Recording to {tracker.csv_path}")
    print(f"[TRACKER] Press Ctrl+C to stop\n")

    last_dashboard = 0.0

    while running:
        case = api.get_case()
        if not case:
            time.sleep(0.5)
            continue

        status = case.get("status", "")
        if status != "ACTIVE":
            if status == "STOPPED":
                print("[TRACKER] Case STOPPED.")
                break
            time.sleep(0.5)
            continue

        tick = int(case.get("tick", 0))
        period = int(case.get("period", 1))

        # Update mock state with real market data
        secs = api.get_securities()
        if secs:
            for sec in secs:
                t = sec.get("ticker", "")
                if t in state._ts:
                    ms = state._ts[t]
                    bid = sec.get("bid")
                    ask = sec.get("ask")
                    if bid and ask:
                        ms.last_mid = (float(bid) + float(ask)) / 2.0
                        ms.last_spread = float(ask) - float(bid)

        snap = tracker.record(tick, period, state, None, None, None, None, None)

        if snap and (time.time() - last_dashboard) > 5.0:
            tracker.print_live_dashboard(snap)
            last_dashboard = time.time()

        time.sleep(1.0)

    tracker.save()
    tracker.print_summary()


if __name__ == "__main__":
    run_standalone()
