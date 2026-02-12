#!/usr/bin/env python3
"""
algo_case_toolkit.py

Two modes:
  1) analyze: read 1 CSV or a folder of ~300 CSV snapshots, compute:
       - clean mid-price time series for SPNG/SMMR/ATMN/WNTR
       - static + rolling correlation
       - beta-to-factor (correlation-aware protection signal)
       - aggregate position limit series extracted from news text

  2) trade: run a correlation-aware market-making bot using the RIT API:
       - quotes both sides with inventory skew
       - uses rolling correlation/factor move to reduce adverse selection
       - aggressively protects aggregate position limit at minute closes
       - avoids high cancel/replace spam (rate limited refresh)

Designed for the RITC 2026 Algorithmic Market Making case constraints:
- 4 tickers: SPNG, SMMR, ATMN, WNTR
- Market orders fee; passive fills receive rebates; max order size 10k
- Aggregate position limit announced via news; penalty assessed at each minute close
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# -----------------------------
# Configuration (safe defaults)
# -----------------------------

TICKERS = ["SPNG", "SMMR", "ATMN", "WNTR"]

# Case economics (used only for prioritization / sizing logic)
# Rebates per share for passive fills (from case package)
PASSIVE_REBATE = {"SPNG": 0.01, "SMMR": 0.02, "ATMN": 0.015, "WNTR": 0.025}
MARKET_FEE = 0.02
MAX_ORDER_SIZE = 10_000
TICK_SIZE = 0.01  # typical in RIT equity cases

# Risk buffers
AGG_LIMIT_SOFT = 0.85   # start throttling + widening
AGG_LIMIT_HARD = 0.95   # go into flatten mode
MINUTE_FLATTEN_WINDOW_SEC = 3  # last seconds of each minute -> flatten inventory

# Quoting cadence (avoid spam)
QUOTE_REFRESH_SEC = 0.40  # cancel/replace at most ~2.5 times/sec total per ticker
MIN_CANCEL_REPLACE_SEC = 0.30

# Correlation model
ROLL_WINDOW = 60  # ticks (~seconds) for rolling corr / factor
EWMA_LAMBDA = 0.94  # for covariance updates (higher = smoother)


# -----------------------------
# Logging
# -----------------------------

def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------
# CSV Reading + Feature Engineering
# -----------------------------

AGG_LIMIT_PATTERNS = [
    re.compile(r"aggregate position limit\s+to\s+([0-9,]+)\s*shares", re.I),
    re.compile(r"aggregate position limit\s+is\s+([0-9,]+)\s*shares", re.I),
    re.compile(r"aggregate position limit\s*[:=]\s*([0-9,]+)", re.I),
]


def parse_aggregate_limit(text: str) -> Optional[int]:
    if not text or not isinstance(text, str):
        return None
    for pat in AGG_LIMIT_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(1).replace(",", "").strip()
            try:
                return int(raw)
            except ValueError:
                return None
    return None


def read_market_csv(path: str) -> pd.DataFrame:
    """
    Reads one market_data CSV snapshot.
    Expected columns (example you uploaded): ticker, last_price, bid_price, ask_price, tick, timestamp, news_*
    Also has dynamic BUY_* and SELL_* depth columns.

    Returns raw dataframe with parsed timestamp.
    """
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["source_file"] = os.path.basename(path)
    return df


def normalize_snapshots(raw: pd.DataFrame) -> pd.DataFrame:
    """
    From raw rows, keep the last update per (tick, ticker),
    compute mid/spread/depth/imbalance, and extract news + agg_limit events.
    """
    needed = {"tick", "ticker"}
    if not needed.issubset(set(raw.columns)):
        raise ValueError(f"CSV missing required columns: {sorted(list(needed - set(raw.columns)))}")

    # Sort so groupby.tail(1) gives the last row per (tick,ticker)
    sort_cols = ["tick"]
    if "timestamp" in raw.columns:
        sort_cols.append("timestamp")
    raw = raw.sort_values(sort_cols, kind="mergesort")

    latest = raw.groupby(["tick", "ticker"], as_index=False).tail(1).copy()

    # Mid & spread
    bid = latest.get("bid_price")
    ask = latest.get("ask_price")
    last = latest.get("last_price")

    # Compute mid using bid/ask when sensible, else last
    latest["mid"] = np.where(
        (bid.notna()) & (ask.notna()) & (ask >= bid) & (bid > 0),
        (bid + ask) / 2.0,
        last,
    )
    latest["spread"] = np.where(
        (bid.notna()) & (ask.notna()) & (ask >= bid),
        (ask - bid),
        np.nan,
    )

    # Depth columns (dynamic)
    buy_cols = [c for c in latest.columns if c.startswith("BUY_")]
    sell_cols = [c for c in latest.columns if c.startswith("SELL_")]
    if buy_cols:
        latest["bid_depth"] = latest[buy_cols].sum(axis=1, skipna=True)
    else:
        latest["bid_depth"] = np.nan
    if sell_cols:
        latest["ask_depth"] = latest[sell_cols].sum(axis=1, skipna=True)
    else:
        latest["ask_depth"] = np.nan

    # Imbalance in [-1, +1]
    bd = latest["bid_depth"].fillna(0.0)
    ad = latest["ask_depth"].fillna(0.0)
    denom = (bd + ad).replace(0.0, np.nan)
    latest["imbalance"] = (bd - ad) / denom

    # News + agg limit
    if "news_body" in latest.columns:
        latest["agg_limit"] = latest["news_body"].apply(parse_aggregate_limit)
    else:
        latest["agg_limit"] = np.nan

    return latest


def load_market_folder(data_dir: str, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load and normalize every CSV in the folder.
    """
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} matching {pattern}")

    frames = []
    for fp in files:
        try:
            raw = read_market_csv(fp)
            norm = normalize_snapshots(raw)
            frames.append(norm)
        except Exception as e:
            logging.warning("Skipping %s due to error: %s", fp, e)

    if not frames:
        raise RuntimeError("No usable CSV files loaded.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["ticker"].isin(TICKERS)].copy()
    all_df.sort_values(["tick", "ticker"], inplace=True)
    return all_df


def build_wide_mid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a wide dataframe indexed by tick with columns = tickers, values = mid.
    Forward fills missing mids.
    """
    wide = df.pivot_table(index="tick", columns="ticker", values="mid", aggfunc="last").sort_index()
    wide = wide.reindex(columns=TICKERS)
    wide = wide.ffill()
    return wide


def compute_returns(wide_mid: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns of mids.
    """
    logp = np.log(wide_mid.replace(0, np.nan))
    rets = logp.diff().fillna(0.0)
    return rets


def correlation_reports(wide_mid: pd.DataFrame, window: int = ROLL_WINDOW) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - static correlation matrix
      - long rolling correlation table (tick, pair, corr)
    """
    rets = compute_returns(wide_mid)
    static_corr = rets.corr()

    # rolling corr pairs
    rolling = rets.rolling(window=window).corr()
    # rolling is MultiIndex (tick, ticker) x ticker; convert to long
    rows = []
    for t in rolling.index.get_level_values(0).unique():
        sub = rolling.loc[t]
        if isinstance(sub, pd.DataFrame):
            for i in TICKERS:
                for j in TICKERS:
                    if i < j:
                        val = sub.loc[i, j]
                        if pd.notna(val):
                            rows.append({"tick": t, "pair": f"{i}-{j}", "corr": float(val)})
    rolling_long = pd.DataFrame(rows)
    return static_corr, rolling_long


def extract_limit_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts aggregate limit announcements from normalized snapshot dataframe.
    Returns tick -> agg_limit, forward-filled.
    """
    lim = df[["tick", "agg_limit"]].dropna().drop_duplicates(subset=["tick"]).sort_values("tick")
    if lim.empty:
        return pd.DataFrame(columns=["tick", "agg_limit"])

    # forward-fill across ticks observed in data
    ticks = pd.DataFrame({"tick": sorted(df["tick"].unique())})
    out = ticks.merge(lim, on="tick", how="left")
    out["agg_limit"] = out["agg_limit"].ffill()
    return out


def estimate_factor_betas(rets: pd.DataFrame) -> pd.Series:
    """
    Simple 1-factor model:
      factor = equal-weight return across tickers
      beta_i = cov(r_i, factor) / var(factor)
    """
    factor = rets.mean(axis=1)
    var_f = float(np.var(factor, ddof=1)) if len(factor) > 2 else 0.0
    betas = {}
    for tkr in TICKERS:
        if var_f <= 1e-12:
            betas[tkr] = 1.0
        else:
            cov = float(np.cov(rets[tkr].values, factor.values, ddof=1)[0, 1])
            betas[tkr] = cov / var_f
    return pd.Series(betas)


# -----------------------------
# RIT API Wrapper (best-effort)
# -----------------------------

class RitClient:
    def __init__(self, host: str, port: int, api_key: str, timeout: float = 0.8):
        self.base = f"{host}:{port}/v1"
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": api_key})
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = self.base + path
        r = self.s.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: Optional[dict] = None, json_body: Optional[dict] = None) -> dict:
        url = self.base + path
        r = self.s.post(url, params=params, json=json_body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_case(self) -> dict:
        return self._get("/case")

    def get_securities(self) -> List[dict]:
        # Typically returns list of securities with bid/ask/last/position
        data = self._get("/securities")
        if isinstance(data, dict) and "securities" in data:
            return data["securities"]
        if isinstance(data, list):
            return data
        return []

    def get_book(self, ticker: str) -> dict:
        # Typical endpoint in many RIT cases
        return self._get("/securities/book", params={"ticker": ticker})

    def get_open_orders(self) -> List[dict]:
        data = self._get("/orders", params={"status": "OPEN"})
        if isinstance(data, dict) and "orders" in data:
            return data["orders"]
        if isinstance(data, list):
            return data
        return []

    def place_limit(self, ticker: str, action: str, qty: int, price: float) -> dict:
        qty = int(max(1, min(MAX_ORDER_SIZE, qty)))
        price = float(price)
        body = {"ticker": ticker, "type": "LIMIT", "quantity": qty, "action": action.upper(), "price": price}
        return self._post("/orders", json_body=body)

    def place_market(self, ticker: str, action: str, qty: int) -> dict:
        qty = int(max(1, min(MAX_ORDER_SIZE, qty)))
        body = {"ticker": ticker, "type": "MARKET", "quantity": qty, "action": action.upper()}
        return self._post("/orders", json_body=body)

    def cancel_order(self, order_id: int) -> dict:
        return self._post("/commands/cancel", params={"id": order_id})

    def cancel_all(self) -> dict:
        return self._post("/commands/cancel", params={"all": 0})

    def get_news(self, since: int = 0) -> List[dict]:
        # Many RIT cases support /news with "since" (tick) param
        data = self._get("/news", params={"since": since})
        if isinstance(data, dict) and "news" in data:
            return data["news"]
        if isinstance(data, list):
            return data
        return []


# -----------------------------
# Trading Model + Bot
# -----------------------------

@dataclass
class QuoteState:
    bid_id: Optional[int] = None
    ask_id: Optional[int] = None
    last_refresh_ts: float = 0.0


class EWMACovModel:
    """
    EWMA covariance model on returns.
    Also stores betas to an equal-weight factor for a simple correlation-aware protection signal.
    """

    def __init__(self, tickers: List[str], lam: float = EWMA_LAMBDA):
        self.tickers = tickers
        self.lam = lam
        self.mu = np.zeros(len(tickers), dtype=float)
        self.cov = np.eye(len(tickers), dtype=float) * 1e-6
        self.last_price = None

    def seed_from_history(self, wide_mid: pd.DataFrame) -> None:
        rets = compute_returns(wide_mid)[self.tickers].values
        if rets.shape[0] < 5:
            return
        self.mu = np.mean(rets, axis=0)
        self.cov = np.cov(rets.T) + np.eye(len(self.tickers)) * 1e-8

    def update_from_prices(self, mid_prices: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """
        Update EWMA on 1-step returns derived from mid prices.
        Returns (r_vec, factor_return).
        """
        p = np.array([mid_prices[t] for t in self.tickers], dtype=float)
        if self.last_price is None:
            self.last_price = p
            return np.zeros(len(self.tickers)), 0.0

        r = np.log(np.maximum(p, 1e-9)) - np.log(np.maximum(self.last_price, 1e-9))
        self.last_price = p

        # EWMA mean/cov update
        self.mu = self.lam * self.mu + (1 - self.lam) * r
        demeaned = (r - self.mu).reshape(-1, 1)
        self.cov = self.lam * self.cov + (1 - self.lam) * (demeaned @ demeaned.T)

        factor = float(np.mean(r))
        return r, factor

    def sigma(self) -> Dict[str, float]:
        s = np.sqrt(np.maximum(np.diag(self.cov), 1e-12))
        return {t: float(s[i]) for i, t in enumerate(self.tickers)}

    def factor_betas(self) -> Dict[str, float]:
        """
        beta_i = cov(i, factor) / var(factor) where factor = equal-weight
        For equal-weight factor: cov(i,f) = mean_j cov(i,j) / n
        var(f) = mean_{i,j} cov(i,j) / n^2
        """
        C = self.cov
        n = C.shape[0]
        cov_if = C.mean(axis=1)  # mean over j
        var_f = float(C.mean()) / (n * n) * (n * n)  # simplify -> just mean(C)
        # Actually for equal weights w=1/n: var_f = w^T C w = mean(C)/n^2 * n^2? No: w^T C w = sum_ij C_ij / n^2 = mean(C).
        var_f = float(C.sum()) / float(n * n)
        if var_f <= 1e-12:
            return {t: 1.0 for t in self.tickers}
        betas = cov_if / var_f
        return {t: float(betas[i]) for i, t in enumerate(self.tickers)}


class MarketMakerBot:
    def __init__(self, client: RitClient, model: EWMACovModel):
        self.c = client
        self.model = model
        self.quote_state: Dict[str, QuoteState] = {t: QuoteState() for t in TICKERS}
        self.last_news_tick = 0
        self.aggregate_limit: Optional[int] = None

    @staticmethod
    def _round_to_tick(x: float) -> float:
        return round(x / TICK_SIZE) * TICK_SIZE

    def _get_positions_and_mids(self) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, Tuple[float, float]]]:
        secs = self.c.get_securities()
        pos = {}
        mids = {}
        bbo = {}
        for s in secs:
            t = s.get("ticker")
            if t not in TICKERS:
                continue
            position = int(s.get("position", 0))
            bid = s.get("bid", s.get("bid_price", None))
            ask = s.get("ask", s.get("ask_price", None))
            last = s.get("last", s.get("last_price", None))
            # Build mid
            if bid is not None and ask is not None and float(ask) >= float(bid) and float(bid) > 0:
                mid = (float(bid) + float(ask)) / 2.0
            else:
                mid = float(last) if last is not None else np.nan
            pos[t] = position
            mids[t] = mid
            bbo[t] = (float(bid) if bid is not None else np.nan, float(ask) if ask is not None else np.nan)
        return pos, mids, bbo

    @staticmethod
    def _agg_position(positions: Dict[str, int]) -> int:
        return int(sum(abs(int(v)) for v in positions.values()))

    @staticmethod
    def _net_position(positions: Dict[str, int]) -> int:
        return int(abs(sum(int(v) for v in positions.values())))

    def _update_limits_from_news(self, current_tick: int) -> None:
        try:
            news = self.c.get_news(since=self.last_news_tick)
        except Exception:
            return

        for item in news:
            # Different RIT versions use different keys
            body = item.get("body") or item.get("news_body") or ""
            headline = item.get("headline") or item.get("news_headline") or ""
            t = item.get("tick", current_tick)

            text = f"{headline}\n{body}"
            lim = parse_aggregate_limit(text)
            if lim is not None:
                self.aggregate_limit = lim
                logging.warning("Updated aggregate limit from news: %s", lim)

            # Track last tick we processed
            if isinstance(t, int) and t > self.last_news_tick:
                self.last_news_tick = t

    def _should_flatten_for_minute_close(self, tick: int) -> bool:
        # In this case, each minute is a "day" close; ticks are typically seconds.
        # So minute close happens when tick % 60 == 0.
        # Flatten in the final few seconds before each multiple-of-60 tick.
        mod = tick % 60
        return mod >= (60 - MINUTE_FLATTEN_WINDOW_SEC)

    def _cancel_if_exists(self, order_id: Optional[int]) -> None:
        if order_id is None:
            return
        try:
            self.c.cancel_order(order_id)
        except Exception:
            pass

    def _place_two_sided_quotes(
        self,
        ticker: str,
        mid: float,
        bid: float,
        ask: float,
        position: int,
        sigma: float,
        beta: float,
        factor_ret: float,
        agg_pos: int,
        agg_limit: int,
        now: float,
    ) -> None:
        """
        Inventory-skewed, correlation-aware quoting:
          - Use factor_ret to shift a "fair" mid to avoid getting picked off on broad moves
          - Use inventory to skew quotes to reduce position
          - Widen / throttle when close to agg limit
        """
        qs = self.quote_state[ticker]

        # Rate limit refreshes
        if (now - qs.last_refresh_ts) < QUOTE_REFRESH_SEC:
            return

        # If no visible spread, assume a minimal one
        if not (math.isfinite(bid) and math.isfinite(ask) and ask >= bid and bid > 0):
            bid = mid - 0.01
            ask = mid + 0.01

        spread_obs = max(ask - bid, 2 * TICK_SIZE)

        # Correlation-aware protection: shift fair mid by beta * factor_ret over a short horizon
        # (This is a micro "don't sell into a rising tape / don't buy into a falling tape" adjustment.)
        horizon = 3.0  # seconds
        fair_mid = mid * math.exp(beta * factor_ret * horizon)

        # Inventory skew (simple Avellaneda-Stoikov style)
        # More inventory => shift reservation price against inventory direction
        inv_scale = max(2000.0, 0.20 * agg_limit)  # scale with limit so skew feels consistent
        reservation = fair_mid - (position / inv_scale) * (sigma * sigma) * 50_000.0

        # Base half-spread: tighter when we want volume & rebates, wider when risk is high
        half = max(TICK_SIZE, 0.5 * spread_obs - TICK_SIZE)

        # Risk throttles near aggregate limit
        soft = int(AGG_LIMIT_SOFT * agg_limit)
        hard = int(AGG_LIMIT_HARD * agg_limit)

        size_base = int(min(2000, max(200, 0.03 * agg_limit)))
        # Prefer bigger size on higher-rebate names
        size = int(size_base * (1.0 + PASSIVE_REBATE[ticker] / 0.02))
        size = max(100, min(size, MAX_ORDER_SIZE))

        if agg_pos >= soft:
            half = max(half, 2 * TICK_SIZE)
            size = int(size * 0.6)

        if agg_pos >= hard:
            # Don't add inventory; quotes become "get-flat" biased
            half = max(half, 3 * TICK_SIZE)
            size = int(size * 0.4)

        # Build quote prices
        bid_px = self._round_to_tick(reservation - half)
        ask_px = self._round_to_tick(reservation + half)

        # Prevent crossing ourselves / weirdness
        if ask_px <= bid_px:
            ask_px = bid_px + TICK_SIZE

        # Inventory-aware: if long, reduce bid aggressiveness & increase ask aggressiveness (and vice versa)
        inv_bias = max(-3, min(3, position / max(1.0, inv_scale / 10.0)))
        bid_px -= inv_bias * TICK_SIZE
        ask_px -= inv_bias * TICK_SIZE

        # Cancel/replace existing quotes (avoid spam; only if "stale enough")
        if (now - qs.last_refresh_ts) >= MIN_CANCEL_REPLACE_SEC:
            self._cancel_if_exists(qs.bid_id)
            self._cancel_if_exists(qs.ask_id)
            qs.bid_id = None
            qs.ask_id = None

            # Place fresh quotes
            try:
                r1 = self.c.place_limit(ticker, "BUY", size, bid_px)
                qs.bid_id = r1.get("order_id") or r1.get("id")
            except Exception:
                qs.bid_id = None

            try:
                r2 = self.c.place_limit(ticker, "SELL", size, ask_px)
                qs.ask_id = r2.get("order_id") or r2.get("id")
            except Exception:
                qs.ask_id = None

            qs.last_refresh_ts = now

    def _flatten_positions(self, positions: Dict[str, int], bbo: Dict[str, Tuple[float, float]], agg_limit: int) -> None:
        """
        Reduce absolute inventory quickly, prioritizing biggest abs positions.
        Uses marketable LIMIT orders (safer than pure market if book is thin).
        """
        # Flatten target: near zero, but keep a tiny cushion to avoid flip-flopping
        items = sorted(positions.items(), key=lambda kv: abs(kv[1]), reverse=True)

        for tkr, p in items:
            if abs(p) < 50:
                continue
            bid, ask = bbo.get(tkr, (np.nan, np.nan))
            if not (math.isfinite(bid) and math.isfinite(ask) and ask >= bid and bid > 0):
                continue

            qty = min(MAX_ORDER_SIZE, max(100, int(abs(p) * 0.6)))
            if p > 0:
                # sell to reduce long; cross slightly to ensure fill
                px = self._round_to_tick(bid - 2 * TICK_SIZE)
                try:
                    self.c.place_limit(tkr, "SELL", qty, px)
                except Exception:
                    pass
            else:
                # buy to reduce short
                px = self._round_to_tick(ask + 2 * TICK_SIZE)
                try:
                    self.c.place_limit(tkr, "BUY", qty, px)
                except Exception:
                    pass

    def run(self) -> None:
        logging.info("Starting bot...")

        # Seed aggregate limit if it appears quickly in news
        try:
            self._update_limits_from_news(0)
        except Exception:
            pass

        while True:
            try:
                case = self.c.get_case()
                tick = int(case.get("tick", 0))
                status = case.get("status", "ACTIVE")
            except Exception as e:
                logging.error("Case fetch failed: %s", e)
                time.sleep(0.2)
                continue

            if status in {"STOPPED", "ENDED", "PAUSED"}:
                logging.warning("Case status=%s; cancelling all and exiting.", status)
                try:
                    self.c.cancel_all()
                except Exception:
                    pass
                return

            # Update dynamic limits from news
            self._update_limits_from_news(tick)

            # If we still don't know the aggregate limit, fall back to a conservative guess,
            # but keep watching news to update (most heats announce it at the start).
            agg_limit = int(self.aggregate_limit) if self.aggregate_limit else 8000

            # Pull positions + prices
            try:
                positions, mids, bbo = self._get_positions_and_mids()
            except Exception as e:
                logging.error("Securities fetch failed: %s", e)
                time.sleep(0.2)
                continue

            if any(not math.isfinite(mids.get(t, np.nan)) for t in TICKERS):
                time.sleep(0.05)
                continue

            # Update correlation model from current prices
            r_vec, factor_ret = self.model.update_from_prices(mids)
            sigmas = self.model.sigma()
            betas = self.model.factor_betas()

            agg_pos = self._agg_position(positions)
            net_pos = self._net_position(positions)

            # Hard protection approaching minute close (each minute)
            if self._should_flatten_for_minute_close(tick) or agg_pos >= int(AGG_LIMIT_HARD * agg_limit):
                logging.info(
                    "Flatten mode | tick=%s agg_pos=%s/%s net=%s",
                    tick, agg_pos, agg_limit, net_pos
                )
                try:
                    self._flatten_positions(positions, bbo, agg_limit)
                except Exception:
                    pass
                time.sleep(0.10)
                continue

            # Normal quoting
            now = time.time()
            for tkr in TICKERS:
                mid = float(mids[tkr])
                bid, ask = bbo[tkr]
                pos = int(positions.get(tkr, 0))
                sigma = float(sigmas.get(tkr, 1e-4))
                beta = float(betas.get(tkr, 1.0))

                self._place_two_sided_quotes(
                    ticker=tkr,
                    mid=mid,
                    bid=bid,
                    ask=ask,
                    position=pos,
                    sigma=sigma,
                    beta=beta,
                    factor_ret=factor_ret,
                    agg_pos=agg_pos,
                    agg_limit=agg_limit,
                    now=now,
                )

            # Small sleep keeps CPU sane and prevents API hammering
            time.sleep(0.05)


# -----------------------------
# CLI
# -----------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    df = load_market_folder(args.data_dir, pattern=args.pattern)
    wide_mid = build_wide_mid(df)
    rets = compute_returns(wide_mid)

    static_corr, rolling_long = correlation_reports(wide_mid, window=args.window)
    betas = estimate_factor_betas(rets[TICKERS])

    limits = extract_limit_series(df)

    print("\n=== Static correlation (log-returns) ===")
    print(static_corr.round(3))

    print("\n=== Factor betas (equal-weight factor) ===")
    print(betas.round(3))

    out_corr = "out_correlations.csv"
    out_roll = "out_rolling_corr_long.csv"
    out_lim = "out_limits.csv"

    static_corr.to_csv(out_corr, index=True)
    rolling_long.to_csv(out_roll, index=False)
    limits.to_csv(out_lim, index=False)

    logging.info("Wrote: %s, %s, %s", out_corr, out_roll, out_lim)


def cmd_trade(args: argparse.Namespace) -> None:
    client = RitClient(args.host, args.port, args.api_key, timeout=args.timeout)

    model = EWMACovModel(TICKERS, lam=EWMA_LAMBDA)

    # Optional seeding from historical CSV folder (recommended)
    if args.data_dir and os.path.isdir(args.data_dir):
        try:
            df = load_market_folder(args.data_dir, pattern=args.pattern)
            wide_mid = build_wide_mid(df)
            model.seed_from_history(wide_mid)
            logging.info("Seeded EWMA model from historical CSV folder: %s", args.data_dir)
        except Exception as e:
            logging.warning("Could not seed from historical CSVs: %s", e)

    bot = MarketMakerBot(client, model)

    # Safety: start by cancelling any leftovers
    try:
        client.cancel_all()
    except Exception:
        pass

    bot.run()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ALGO Market Making CSV toolkit + bot")
    sub = p.add_subparsers(dest="mode", required=True)

    # analyze
    pa = sub.add_parser("analyze", help="Analyze correlations + extract limits from CSV folder")
    pa.add_argument("--data-dir", required=True, help="Folder containing the market_data_*.csv snapshots")
    pa.add_argument("--pattern", default="*.csv", help="Glob pattern (default: *.csv)")
    pa.add_argument("--window", type=int, default=ROLL_WINDOW, help="Rolling correlation window in ticks")
    pa.add_argument("--log-level", default="INFO")
    pa.set_defaults(func=cmd_analyze)

    # trade
    pt = sub.add_parser("trade", help="Run correlation-aware market making bot (RIT API)")
    pt.add_argument("--host", default="http://localhost", help="RIT host (default http://localhost)")
    pt.add_argument("--port", type=int, default=9998, help="RIT port (default 9998)")
    pt.add_argument("--api-key", required=True, help="API key")
    pt.add_argument("--timeout", type=float, default=0.8, help="HTTP timeout seconds")
    pt.add_argument("--data-dir", default="", help="(Optional) folder of historical CSVs to seed correlations")
    pt.add_argument("--pattern", default="*.csv", help="Glob pattern (default: *.csv)")
    pt.add_argument("--log-level", default="INFO")
    pt.set_defaults(func=cmd_trade)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logger(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
