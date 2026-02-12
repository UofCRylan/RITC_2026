from __future__ import annotations
from dataclasses import dataclass, field
import time
import os
import signal
import sys
import re
import atexit
from typing import Any, Optional, Dict, Tuple, List
import requests
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from tracker import PerformanceTracker

API_URL = "http://localhost:9999/v1"
API_KEY = "8ZTU2PA4"#os.environ.get("RIT_API_KEY", "YOUR_KEY_HERE")


class RITClient:
    """
    Thin, rate-limit-aware wrapper around the RIT Client REST API (localhost:9999/v1).
    """

    def __init__(self, api_url: str = API_URL, api_key: str = API_KEY, timeout: float = 0.5):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        # API_KEY must be a string
        self.session.headers.update({"X-API-Key": api_key})

    # ---------- core request helpers ----------

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        max_retries: int = 3,
        retry_backoff: float = 0.02,
        verbose: bool = False,
    ) -> Optional[Any]:
        """
        Performs an HTTP request with basic handling for:
          - 429 rate limiting: sleep per Retry-After header or JSON 'wait'
          - timeouts / transient errors: retry a few times
        Returns parsed JSON (dict/list) or None.
        """
        endpoint = endpoint.lstrip("/")
        url = f"{self.api_url}/{endpoint}"

        for attempt in range(max_retries + 1):
            try:
                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    timeout=self.timeout,
                )

                # Auth error - don't retry much
                if resp.status_code == 401:
                    if verbose:
                        print(f"[401 Unauthorized] Check API key. Body: {resp.text}")
                    return None

                # Rate limit handling
                if resp.status_code == 429:
                    wait_s = 0.0

                    # Prefer Retry-After header
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            wait_s = float(ra)
                        except ValueError:
                            wait_s = 0.0

                    # Also check JSON 'wait' field
                    try:
                        data = resp.json()
                        if isinstance(data, dict) and "wait" in data:
                            wait_s = max(wait_s, float(data["wait"]))
                    except Exception:
                        pass

                    if attempt >= max_retries:
                        if verbose:
                            print(f"[429 Rate limited] Gave up after retries. wait={wait_s}")
                        return None

                    # tiny backoff so you're not slamming exactly at the boundary
                    time.sleep(wait_s + retry_backoff)
                    continue

                # Other non-OK responses
                if not (200 <= resp.status_code < 300):
                    if verbose:
                        print(f"[{resp.status_code}] {method} {endpoint} failed: {resp.text}")
                    return None

                # Parse JSON
                try:
                    return resp.json()
                except Exception as e:
                    if verbose:
                        print(f"[JSON parse error] {e} | Raw: {resp.text[:200]}")
                    return None

            except requests.exceptions.Timeout:
                if attempt >= max_retries:
                    if verbose:
                        print(f"[Timeout] {method} {endpoint} failed after retries.")
                    return None
                time.sleep(retry_backoff * (attempt + 1))
            except requests.exceptions.RequestException as e:
                if attempt >= max_retries:
                    if verbose:
                        print(f"[RequestException] {e}")
                    return None
                time.sleep(retry_backoff * (attempt + 1))

        return None

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kw) -> Optional[Any]:
        return self._request("GET", endpoint, params=params, **kw)

    def post(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kw) -> Optional[Any]:
        return self._request("POST", endpoint, params=params, **kw)

    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kw) -> Optional[Any]:
        return self._request("DELETE", endpoint, params=params, **kw)

    # ---------- convenience wrappers  ----------

    def get_case(self) -> Optional[dict]:
        return self.get("case")

    def get_trader(self) -> Optional[dict]:
        return self.get("trader")

    def get_limits(self) -> Optional[list]:
        return self.get("limits")

    def get_securities(self, ticker: Optional[str] = None) -> Optional[list]:
        params = {"ticker": ticker} if ticker else None
        return self.get("securities", params=params)

    def get_book(self, ticker: str, limit: int = 20) -> Optional[dict]:
        return self.get("securities/book", params={"ticker": ticker, "limit": limit})

    def get_history(self, ticker: str, period: Optional[int] = None, limit: Optional[int] = None) -> Optional[list]:
        params: Dict[str, Any] = {"ticker": ticker}
        if period is not None:
            params["period"] = period
        if limit is not None:
            params["limit"] = limit
        return self.get("securities/history", params=params)

    def get_tas(
        self,
        ticker: str,
        after: Optional[int] = None,
        period: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Optional[list]:
        params: Dict[str, Any] = {"ticker": ticker}
        if after is not None:
            params["after"] = after
        if period is not None:
            params["period"] = period
        if limit is not None:
            params["limit"] = limit
        return self.get("securities/tas", params=params)

    def get_orders(self, status: str = "OPEN") -> Optional[list]:
        return self.get("orders", params={"status": status})

    def post_order(
        self,
        *,
        ticker: str,
        side: str,              # "BUY" or "SELL"
        qty: float,
        order_type: str,        # "LIMIT" or "MARKET"
        price: Optional[float] = None,
        dry_run: Optional[int] = None,
        wait: bool = False,
    ) -> Optional[dict]:
        """
        POST /orders
        For LIMIT, price is required.
        For MARKET, price is ignored; dry_run optional (0/1).
        """
        order_type = order_type.upper()
        side = side.upper()

        params: Dict[str, Any] = {
            "ticker": ticker,
            "type": order_type,
            "quantity": qty,
            "action": side,
        }
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires price")
            params["price"] = price
        else:
            if dry_run is not None:
                params["dry_run"] = dry_run

        # wait=True will auto-handle 429 with sleeps
        return self.post("orders", params=params, max_retries=6 if wait else 3)

    def cancel_all(self) -> Optional[dict]:
        return self.post("commands/cancel", params={"all": 1})

    def cancel_ticker(self, ticker: str) -> Optional[dict]:
        return self.post("commands/cancel", params={"ticker": ticker})

    def cancel_query(self, query: str) -> Optional[dict]:
        return self.post("commands/cancel", params={"query": query})

    def delete_order(self, order_id: int) -> Optional[dict]:
        return self.delete(f"orders/{order_id}")

    # ---------- news & tenders ----------

    def get_news(self, since: Optional[int] = None, limit: Optional[int] = None) -> Optional[list]:
        params: Dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        return self.get("news", params=params or None)

    def get_tenders(self) -> Optional[list]:
        return self.get("tenders")

    def accept_tender(self, tender_id: int, price: Optional[float] = None) -> Optional[dict]:
        params: Dict[str, Any] = {}
        if price is not None:
            params["price"] = price
        return self.post(f"tenders/{tender_id}", params=params or None)

    def decline_tender(self, tender_id: int) -> Optional[dict]:
        return self.delete(f"tenders/{tender_id}")

    # ---------- small parsing helpers ----------

    @staticmethod
    def top_of_book(book_json: Optional[dict]) -> Optional[tuple[float, float, float, float]]:
        """
        Returns (best_bid_price, best_bid_qty, best_ask_price, best_ask_qty) or None.
        Swagger uses 'bid'/'ask' (singular); try both for safety.
        """
        if not book_json:
            return None

        bids = book_json.get("bids") or book_json.get("bid") or []
        asks = book_json.get("asks") or book_json.get("ask") or []
        if not bids or not asks:
            return None

        best_bid = bids[0]
        best_ask = asks[0]

        return (
            float(best_bid["price"]),
            float(best_bid.get("quantity", 0)),
            float(best_ask["price"]),
            float(best_ask.get("quantity", 0)),
        )

    @staticmethod
    def mid_and_spread(tob: tuple[float, float, float, float]) -> tuple[float, float]:
        bid_p, _, ask_p, _ = tob
        return (bid_p + ask_p) / 2.0, (ask_p - bid_p)




@dataclass
class OrderRef:
    bid_id: Optional[int] = None
    ask_id: Optional[int] = None
    bid_px: Optional[float] = None
    ask_px: Optional[float] = None
    bid_qty: int = 0
    ask_qty: int = 0

# holds state for a single ticker
@dataclass
class TickerState:
    ticker: str

    # Timing / pacing
    last_case_tick: int = -1
    last_refresh_ts: float = 0.0  # time.time()
    min_refresh_interval: float = 0.10  # seconds (was 0.20, reduced for speed)

    # Last seen market
    last_best_bid: Optional[float] = None
    last_best_ask: Optional[float] = None
    last_mid: Optional[float] = None
    last_spread: Optional[float] = None

    # Inventory / risk
    position: int = 0
    target_pos: int = 0
    max_abs_pos: int = 0  # optional: set from limits or your own cap

    # Order tracking
    working: OrderRef = field(default_factory=OrderRef)

    # Momentum / volatility (updated by trackers)
    momentum: float = 0.0        # rolling EMA return, positive = trending up
    volatility: float = 0.0      # rolling realized vol (std of returns)

    # Diagnostics / counters
    num_posts: int = 0
    num_cancels: int = 0
    num_fills: int = 0
    last_error: Optional[str] = None

    # Fill-rate tracking (for auto spread adjustment)
    fill_count_window: int = 0      # fills in current window
    post_count_window: int = 0      # posts in current window
    fill_rate: float = 0.0          # rolling fill rate (0..1)
    fill_rate_window_start: float = 0.0  # time.time() of window start

    # function to check if we should refresh data
    def should_refresh(self, current_case_tick: int) -> bool:
        now = time.time()
        if current_case_tick != self.last_case_tick:
            return True
        if (now - self.last_refresh_ts) >= self.min_refresh_interval:
            return True
        return False
    
    def mark_refreshed(self, current_case_tick: int) -> None:
        self.last_case_tick = current_case_tick
        self.last_refresh_ts = time.time()


# holds state for all tickers
@dataclass
class BotState:
    tickers: List[str]
    per_ticker: Dict[str, TickerState] = field(default_factory=dict)

    # Aggregate position limit (announced via news at start of each heat).
    # This is DIFFERENT from gross/net trading limits.
    # Penalty: $10/share for every share above this limit, assessed at each market close (every minute).
    # Default to a conservative value; will be updated when we parse the news announcement.
    aggregate_limit: int = 9000
    aggregate_limit_parsed: bool = False  # True once we've read it from news

    def __post_init__(self):
        for t in self.tickers:
            self.per_ticker[t] = TickerState(ticker=t)

    def s(self, ticker: str) -> TickerState:
        return self.per_ticker[ticker]
    


@dataclass
class CorrelationTracker:
    tickers: List[str]
    window: int = 100
    mids: Dict[str, deque] = field(default_factory=dict)

    def __post_init__(self):
        for t in self.tickers:
            self.mids[t] = deque(maxlen=self.window)

    def update_mid(self, ticker: str, mid: float) -> None:
        self.mids[ticker].append(float(mid))

    def ready(self, min_points: int = 10) -> bool:
        return all(len(self.mids[t]) >= min_points for t in self.tickers)

    @staticmethod
    def _returns(series: List[float]) -> List[float]:
        rets = []
        for i in range(1, len(series)):
            prev = series[i - 1]
            curr = series[i]
            if prev == 0:
                rets.append(0.0)
            else:
                rets.append((curr - prev) / abs(prev))
        return rets

    @staticmethod
    def _corr(x: List[float], y: List[float]) -> float:
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        x = x[-n:]
        y = y[-n:]
        mx = sum(x) / n
        my = sum(y) / n
        cov = 0.0
        vx = 0.0
        vy = 0.0
        for i in range(n):
            dx = x[i] - mx
            dy = y[i] - my
            cov += dx * dy
            vx += dx * dx
            vy += dy * dy
        if vx <= 0 or vy <= 0:
            return 0.0
        return cov / math.sqrt(vx * vy)

    def corr_matrix(self) -> Dict[Tuple[str, str], float]:
        """Returns dict keyed by (ticker_i, ticker_j) -> correlation (uses returns)."""
        rets: Dict[str, List[float]] = {}
        for t in self.tickers:
            rets[t] = self._returns(list(self.mids[t]))
        out: Dict[Tuple[str, str], float] = {}
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                a, b = self.tickers[i], self.tickers[j]
                out[(a, b)] = 1.0 if a == b else self._corr(rets[a], rets[b])
        return out


# ---------------------------------------------------------------------------
#  MOMENTUM TRACKER – EMA-based trend detection
# ---------------------------------------------------------------------------

@dataclass
class MomentumTracker:
    """
    Tracks per-ticker momentum using an EMA of returns.
    Positive momentum = price trending up, negative = trending down.
    """
    tickers: List[str]
    fast_alpha: float = 0.3    # EMA smoothing for fast signal (reacts quickly)
    slow_alpha: float = 0.05   # EMA smoothing for slow signal (trend confirmation)
    prev_mids: Dict[str, float] = field(default_factory=dict)
    fast_ema: Dict[str, float] = field(default_factory=dict)
    slow_ema: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for t in self.tickers:
            self.prev_mids[t] = 0.0
            self.fast_ema[t] = 0.0
            self.slow_ema[t] = 0.0

    def update(self, ticker: str, mid: float) -> float:
        """
        Feed a new mid price. Returns the current momentum signal.
        Momentum = fast_ema - slow_ema of returns.
        Positive = trending up, Negative = trending down.
        """
        prev = self.prev_mids[ticker]
        self.prev_mids[ticker] = mid

        if prev <= 0:
            return 0.0

        ret = (mid - prev) / prev  # simple return

        # Update EMAs
        self.fast_ema[ticker] = self.fast_alpha * ret + (1 - self.fast_alpha) * self.fast_ema[ticker]
        self.slow_ema[ticker] = self.slow_alpha * ret + (1 - self.slow_alpha) * self.slow_ema[ticker]

        # Momentum = difference between fast and slow (MACD-style)
        return self.fast_ema[ticker] - self.slow_ema[ticker]

    def get_momentum(self, ticker: str) -> float:
        return self.fast_ema.get(ticker, 0.0) - self.slow_ema.get(ticker, 0.0)

    def get_trend_direction(self, ticker: str) -> float:
        """Returns the slow EMA as a trend direction indicator."""
        return self.slow_ema.get(ticker, 0.0)

    def reset(self, ticker: str = "") -> None:
        """Reset momentum state. Call at day boundaries to avoid stale signals."""
        if ticker:
            self.fast_ema[ticker] = 0.0
            self.slow_ema[ticker] = 0.0
            self.prev_mids[ticker] = 0.0
        else:
            for t in self.fast_ema:
                self.fast_ema[t] = 0.0
                self.slow_ema[t] = 0.0
                self.prev_mids[t] = 0.0


# ---------------------------------------------------------------------------
#  VOLATILITY TRACKER – rolling realized volatility
# ---------------------------------------------------------------------------

@dataclass
class VolatilityTracker:
    """
    Tracks per-ticker realized volatility using a rolling window of returns.
    Used to dynamically widen/tighten spreads.
    """
    tickers: List[str]
    window: int = 30  # number of observations for vol calculation
    returns: Dict[str, deque] = field(default_factory=dict)
    prev_mids: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for t in self.tickers:
            self.returns[t] = deque(maxlen=self.window)
            self.prev_mids[t] = 0.0

    def update(self, ticker: str, mid: float) -> float:
        """Feed a new mid. Returns current realized vol (std of returns)."""
        prev = self.prev_mids[ticker]
        self.prev_mids[ticker] = mid

        if prev <= 0:
            return 0.0

        ret = (mid - prev) / prev
        self.returns[ticker].append(ret)
        return self.get_volatility(ticker)

    def get_volatility(self, ticker: str) -> float:
        """Standard deviation of recent returns."""
        rets = self.returns.get(ticker)
        if not rets or len(rets) < 5:
            return 0.0
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / n
        return math.sqrt(var)

    def get_vol_ratio(self, ticker: str, baseline_vol: float = 0.003) -> float:
        """
        Returns vol / baseline.  >1 means more volatile than normal.
        Used as a spread multiplier.
        When we have no data, return 1.0 (assume baseline, don't shrink spread).
        """
        v = self.get_volatility(ticker)
        if baseline_vol <= 0:
            return 1.0
        if v <= 0:
            return 1.0  # no data yet — assume baseline, never shrink spread
        return max(1.0, v / baseline_vol)  # floor at 1.0x (vol only ever WIDENS)

    def reset(self, ticker: str = "") -> None:
        """Reset vol state. Call at day boundaries to avoid stale regime-change outliers."""
        if ticker:
            self.returns[ticker].clear()
            self.prev_mids[ticker] = 0.0
        else:
            for t in self.returns:
                self.returns[t].clear()
                self.prev_mids[t] = 0.0


# ---------------------------------------------------------------------------
#  NEWS PARSER  – keyword-driven fair-value adjustment
# ---------------------------------------------------------------------------

# Positive / negative keyword banks.  Each maps to a magnitude bucket.
_POS_KEYWORDS: Dict[str, float] = {
    "surge": 0.6, "surges": 0.6, "soar": 0.6, "soars": 0.6,
    "jump": 0.5, "jumps": 0.5, "rally": 0.5, "rallies": 0.5,
    "boom": 0.5, "booming": 0.5, "skyrocket": 0.7,
    "rise": 0.3, "rises": 0.3, "rising": 0.3,
    "gain": 0.3, "gains": 0.3, "increase": 0.3, "increases": 0.3,
    "up": 0.2, "higher": 0.2, "bullish": 0.4,
    "strong": 0.3, "positive": 0.3, "growth": 0.3,
    "outperform": 0.4, "upgrade": 0.4, "beat": 0.3, "beats": 0.3,
    "record": 0.3, "high": 0.2, "recover": 0.3, "recovery": 0.3,
    "optimism": 0.3, "optimistic": 0.3, "exceed": 0.3, "exceeds": 0.3,
    "profit": 0.3, "profitable": 0.3, "dividend": 0.2,
}

_NEG_KEYWORDS: Dict[str, float] = {
    "crash": 0.7, "crashes": 0.7, "plunge": 0.6, "plunges": 0.6,
    "plummet": 0.6, "plummets": 0.6, "collapse": 0.6,
    "drop": 0.4, "drops": 0.4, "fall": 0.4, "falls": 0.4, "falling": 0.4,
    "decline": 0.4, "declines": 0.4, "decrease": 0.3, "decreases": 0.3,
    "down": 0.2, "lower": 0.2, "bearish": 0.4,
    "weak": 0.3, "negative": 0.3, "loss": 0.4, "losses": 0.4,
    "miss": 0.3, "misses": 0.3, "downgrade": 0.4,
    "sell-off": 0.5, "selloff": 0.5, "recession": 0.5,
    "risk": 0.2, "warning": 0.3, "concern": 0.2, "fear": 0.3,
    "cut": 0.3, "cuts": 0.3, "slump": 0.5, "tumble": 0.5,
    "low": 0.2, "underperform": 0.4, "penalty": 0.3,
}

# Percentage patterns, e.g. "up 5%", "+3.2%", "fell 10 percent"
_PCT_PATTERN = re.compile(
    r"(?:up|rose|gain(?:ed)?|increase[ds]?|jump(?:ed)?|surge[ds]?)\s+(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_PCT_NEG_PATTERN = re.compile(
    r"(?:down|fell|drop(?:ped)?|decline[ds]?|decrease[ds]?|plunge[ds]?|lost?)\s+(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def parse_news_sentiment(headline: str, body: str) -> float:
    """
    Returns a sentiment score in roughly [-1, +1].
    Positive = bullish, Negative = bearish.
    Uses keyword matching + percentage extraction.
    """
    text = f"{headline} {body}".lower()
    words = re.findall(r"[a-z\-]+", text)

    pos_score = 0.0
    neg_score = 0.0
    for w in words:
        if w in _POS_KEYWORDS:
            pos_score += _POS_KEYWORDS[w]
        if w in _NEG_KEYWORDS:
            neg_score += _NEG_KEYWORDS[w]

    # Extract explicit percentages
    for m in _PCT_PATTERN.finditer(text):
        pct = float(m.group(1))
        pos_score += min(pct / 10.0, 1.0)  # cap contribution
    for m in _PCT_NEG_PATTERN.finditer(text):
        pct = float(m.group(1))
        neg_score += min(pct / 10.0, 1.0)

    total = pos_score + neg_score
    if total == 0:
        return 0.0
    # net sentiment scaled to [-1, 1]
    raw = (pos_score - neg_score) / max(total, 1.0)
    return max(-1.0, min(1.0, raw))


def extract_mentioned_tickers(text: str, known_tickers: List[str]) -> List[str]:
    """Return which of the known tickers are explicitly mentioned in the text."""
    upper = text.upper()
    return [t for t in known_tickers if t in upper]


# Pattern to extract aggregate position limit from news announcements
_AGG_LIMIT_PATTERN = re.compile(
    r"aggregate\s+position\s+limit[^\d]*(\d[\d,]*)",
    re.IGNORECASE,
)
# Fallback pattern: "position limit of X shares" or "limit is X"
_AGG_LIMIT_PATTERN2 = re.compile(
    r"position\s+limit[^\d]*(\d[\d,]*)",
    re.IGNORECASE,
)


@dataclass
class NewsTracker:
    """Polls /news, parses sentiment, and provides per-ticker fair-value adjustments."""
    known_tickers: List[str]
    last_news_id: int = 0

    # Per-ticker accumulated sentiment adjustment (decays over time)
    adjustments: Dict[str, float] = field(default_factory=dict)

    # How much a sentiment=1.0 shifts fair value (in dollars)
    max_shift_per_event: float = 0.15
    # Decay factor applied each tick (0 = instant decay, 1 = no decay)
    decay: float = 0.85

    # Parsed aggregate position limit (None until found in news)
    parsed_aggregate_limit: Optional[int] = None

    def __post_init__(self):
        for t in self.known_tickers:
            self.adjustments.setdefault(t, 0.0)

    def _try_parse_aggregate_limit(self, text: str) -> Optional[int]:
        """Try to extract aggregate position limit from news text."""
        for pattern in [_AGG_LIMIT_PATTERN, _AGG_LIMIT_PATTERN2]:
            m = pattern.search(text)
            if m:
                val = int(m.group(1).replace(",", ""))
                if 1000 <= val <= 100000:  # sanity check
                    return val
        return None

    def poll(self, api: RITClient) -> None:
        """Fetch new news items and update adjustments."""
        news = api.get_news(since=self.last_news_id, limit=50)
        if not news:
            return
        for item in news:
            nid = int(item.get("news_id", 0))
            if nid <= self.last_news_id:
                continue
            self.last_news_id = nid

            headline = item.get("headline", "")
            body = item.get("body", "")
            full_text = f"{headline} {body}"

            # --- Try to parse aggregate position limit from this news ---
            agg_limit = self._try_parse_aggregate_limit(full_text)
            if agg_limit is not None:
                self.parsed_aggregate_limit = agg_limit
                print(f"  [NEWS] *** AGGREGATE POSITION LIMIT = {agg_limit:,} shares ***")

            sentiment = parse_news_sentiment(headline, body)
            if sentiment == 0.0:
                continue

            shift = sentiment * self.max_shift_per_event

            # If specific tickers are mentioned, only adjust those
            mentioned = extract_mentioned_tickers(full_text, self.known_tickers)
            targets = mentioned if mentioned else self.known_tickers

            for t in targets:
                self.adjustments[t] = self.adjustments.get(t, 0.0) + shift

            if abs(sentiment) > 0.1:
                tgt_str = ",".join(targets)
                print(f"  [NEWS] id={nid} sentiment={sentiment:+.2f} shift={shift:+.3f} -> {tgt_str} | {headline[:60]}")

    def decay_adjustments(self) -> None:
        """Call once per tick to decay old adjustments toward zero."""
        for t in self.known_tickers:
            self.adjustments[t] *= self.decay

    def get_adjustment(self, ticker: str) -> float:
        return self.adjustments.get(ticker, 0.0)


# ---------------------------------------------------------------------------
#  QUOTING ENGINE
# ---------------------------------------------------------------------------

@dataclass
class QuotingParams:
    """Tunable parameters for the market-making strategy."""
    # Base half-spread in dollars (each side offset from fair value)
    half_spread: float = 0.05

    # Order size per side
    order_size: int = 1000

    # Inventory skew: how much to shift quotes per unit of position
    # e.g. 0.001 means 1 cent shift per 10 shares of inventory
    inventory_skew_per_share: float = 0.001

    # Max inventory skew in dollars (caps the shift)
    max_inventory_skew: float = 0.15

    # Minimum distance (in $) the market must move before we cancel+re-quote
    requote_threshold: float = 0.02

    # When aggregate position is this fraction of limit, start widening spread
    limit_warning_pct: float = 0.70

    # Extra spread added when at the limit warning level (scales linearly to 1.0)
    limit_extra_spread: float = 0.10

    # Size reduction when near limit (fraction of normal size)
    limit_size_reduction: float = 0.25

    # At this fraction of limit, stop quoting the side that increases exposure
    limit_hard_pct: float = 0.90

    # --- Momentum / adverse-selection protection ---
    # How much momentum shifts fair value (in $ per unit momentum signal)
    momentum_fv_shift: float = 2.0

    # Extra spread added per unit of momentum (protects against trends)
    momentum_spread_add: float = 1.5

    # Momentum threshold below which we ignore the signal (noise filter)
    momentum_dead_zone: float = 0.0005

    # --- Volatility-adaptive spread ---
    # Spread multiplied by vol_ratio: vol/baseline.  1.0 = normal, 2.0 = double spread
    vol_spread_enabled: bool = True
    vol_baseline: float = 0.003  # baseline per-tick return std (~0.3%)
    vol_max_multiplier: float = 3.0  # cap vol multiplier

    # --- Market-close flattening ---
    # Ticks per "trading day" (1 minute = 60 ticks at 1 tick/sec)
    ticks_per_day: int = 60
    # Start flattening this many ticks before market close
    flatten_start_ticks_before_close: int = 8
    # Hard-flatten (aggressive passive limit orders) this many ticks before close
    flatten_hard_ticks_before_close: int = 3
    # Max shares per passive limit order per ticker per flatten cycle
    flatten_market_order_max: int = 2000

    # --- Per-ticker sizing by rebate ---
    # Rebate per share for each ticker (used to scale order size)
    ticker_rebates: Dict[str, float] = field(default_factory=lambda: {
        "SPNG": 0.01, "SMMR": 0.02, "ATMN": 0.015, "WNTR": 0.025,
    })
    # If True, scale order_size proportional to rebate (higher rebate = larger size)
    rebate_sizing_enabled: bool = True
    # Base rebate used for normalization (SPNG's rebate)
    rebate_baseline: float = 0.01

    # --- Adaptive spread from actual book ---
    # If True, use the actual market spread to set our spread (rather than fixed half_spread)
    adaptive_spread_enabled: bool = True
    # Our spread as a fraction of the market spread (0.8 = slightly inside)
    adaptive_spread_fraction: float = 0.80
    # Floor: never go below this half-spread even if market is very tight
    adaptive_spread_floor: float = 0.02
    # Ceiling: never go above this even if market is very wide
    adaptive_spread_ceiling: float = 0.20

    # --- News blackout window ---
    # At each market open, news drops and causes regime changes ($0.50-$1.00 price moves).
    # Instead of widening spread (still gets picked off), go COMPLETELY DARK:
    # cancel all orders, observe the price move, resume quoting once settled.
    news_blackout_ticks: int = 8  # don't quote for first 8 ticks of each trading day

    # --- Net limit awareness ---
    # If net position (sum of signed positions) exceeds this fraction of net_limit,
    # bias quoting to rebalance toward zero net
    net_limit_warning_pct: float = 0.60
    # Extra skew per share of net position (added on top of per-ticker skew)
    net_skew_per_share: float = 0.0003

    # --- Per-ticker position cap ---
    # Max absolute position per ticker as fraction of aggregate limit
    # With 4 tickers and limit=9000, each gets max 2250 (25% each)
    per_ticker_limit_fraction: float = 0.25

    # --- Fill-rate auto-tuning ---
    # Target fill rate (fraction of posted quotes that get filled)
    fill_rate_target: float = 0.30
    # Window in seconds for measuring fill rate
    fill_rate_window_secs: float = 30.0
    # Max spread adjustment from fill-rate feedback (in $)
    fill_rate_max_adjust: float = 0.03


def compute_fair_value(
    mid: float,
    news_adj: float,
    momentum: float,
    params: QuotingParams,
) -> float:
    """
    Fair value = mid + news_adjustment + momentum_shift.
    Momentum shifts FV in the direction of the trend so we don't
    provide stale liquidity into a moving market.
    """
    fv = mid + news_adj

    # Momentum adjustment: VERY small shift only.
    # Large shifts cause the bot to trend-follow (sell into drops, buy into spikes)
    # which is the OPPOSITE of market-making. Keep this minimal.
    if abs(momentum) > params.momentum_dead_zone:
        fv += momentum * params.momentum_fv_shift

    return fv


def compute_quotes(
    fair_value: float,
    position: int,
    params: QuotingParams,
    agg_position_ratio: float,  # aggregate_position / aggregate_limit (0..1+)
    side_increasing_long: bool,  # True if this ticker's position > 0
    momentum: float = 0.0,      # momentum signal for this ticker
    vol_ratio: float = 1.0,     # volatility / baseline (>1 = more volatile)
    ticker: str = "",           # ticker symbol (for rebate sizing)
    market_spread: float = 0.0, # actual observed bid-ask spread
    net_position: int = 0,      # net position across ALL tickers (signed sum)
    net_limit: int = 25000,     # net trading limit
    ticks_into_day: int = 30,   # how many ticks since last market open
    fill_rate: float = 0.30,    # recent fill rate for this ticker
    agg_limit: int = 10000,     # aggregate position limit (for per-ticker cap)
) -> Tuple[Optional[float], Optional[float], int, int]:
    """
    Returns (bid_price, ask_price, bid_size, ask_size).
    Any price may be None if we should NOT quote that side.

    Incorporates:
    - Inventory skew (shifts both quotes against position)
    - Net-position skew (balances across all tickers)
    - Momentum (widens spread when trending)
    - Volatility (widens spread when volatile)
    - Adaptive spread from actual book
    - Pre-news spread widening
    - Per-ticker rebate sizing
    - Fill-rate auto-tuning
    - Aggregate limit guardrails
    """
    # --- Per-ticker sizing by rebate ---
    base_size = params.order_size
    if params.rebate_sizing_enabled and ticker in params.ticker_rebates:
        rebate = params.ticker_rebates[ticker]
        # Scale size proportional to rebate, but CAP at 1.5x to avoid position blowups
        size_mult = rebate / max(params.rebate_baseline, 0.001)
        size_mult = min(size_mult, 1.5)  # hard cap: never more than 1.5x base
        base_size = max(100, int(params.order_size * size_mult))

    bid_size = base_size
    ask_size = base_size

    # --- Adaptive spread from actual book ---
    if params.adaptive_spread_enabled and market_spread > 0:
        # Set our half-spread as a fraction of the observed market spread
        adaptive_half = (market_spread / 2.0) * params.adaptive_spread_fraction
        adaptive_half = max(params.adaptive_spread_floor, min(params.adaptive_spread_ceiling, adaptive_half))
        half = adaptive_half
    else:
        half = params.half_spread

    # --- Volatility-adaptive spread ---
    if params.vol_spread_enabled:
        clamped_vol = min(vol_ratio, params.vol_max_multiplier)
        half *= clamped_vol

    # --- News blackout (handled in main loop, not here) ---
    # The main loop cancels all orders and stops quoting entirely during
    # the first N ticks of each trading day. No spread widening needed here.

    # --- Momentum spread widening ---
    abs_mom = abs(momentum)
    if abs_mom > params.momentum_dead_zone:
        half += abs_mom * params.momentum_spread_add

    # --- Fill-rate auto-tuning ---
    # If fill rate is too high, we're too tight → widen spread
    # If fill rate is low, do NOT tighten — the market may be quiet/toxic
    # Only ever WIDEN, never tighten from this signal
    if fill_rate > params.fill_rate_target:
        fill_excess = fill_rate - params.fill_rate_target
        fill_adjust = min(fill_excess * params.fill_rate_max_adjust / 0.3, params.fill_rate_max_adjust)
        half += fill_adjust

    # Floor the half-spread at 1 cent
    half = max(0.01, half)

    # --- Inventory skew (per-ticker) ---
    # Skew shifts BOTH quotes against our position to encourage mean-reversion.
    # CRITICAL: skew must NEVER exceed half-spread, or the reducing side
    # crosses fair value and every fill on that side is a guaranteed loss.
    # Cap skew at 50% of half-spread so we always maintain edge on both sides.
    raw_skew = position * params.inventory_skew_per_share
    max_allowed_skew = min(params.max_inventory_skew, half * 0.5)
    skew = max(-max_allowed_skew, min(max_allowed_skew, raw_skew))

    # --- Net-position skew (cross-ticker rebalancing) ---
    if abs(net_position) > net_limit * params.net_limit_warning_pct:
        net_skew = net_position * params.net_skew_per_share
        net_skew = max(-params.max_inventory_skew, min(params.max_inventory_skew, net_skew))
        skew += net_skew

    # --- Aggregate limit guardrails ---
    if agg_position_ratio > params.limit_warning_pct:
        pct_into_warning = min(1.0, (agg_position_ratio - params.limit_warning_pct) / (1.0 - params.limit_warning_pct))
        half += params.limit_extra_spread * pct_into_warning
        size_mult = 1.0 - (params.limit_size_reduction * pct_into_warning)
        bid_size = max(100, int(base_size * size_mult))
        ask_size = max(100, int(base_size * size_mult))

    bid_px = round(fair_value - half - skew, 2)
    ask_px = round(fair_value + half - skew, 2)

    # --- Per-ticker position cap ---
    # Prevent any single ticker from using more than its share of the aggregate limit.
    # E.g. with limit=14000 and 4 tickers, each gets max 3500 (25%).
    per_ticker_max = int(agg_limit * params.per_ticker_limit_fraction)
    if per_ticker_max > 0:
        abs_pos = abs(position)
        room = max(0, per_ticker_max - abs_pos)

        if abs_pos >= per_ticker_max:
            # HARD CAP: only allow reducing side, block increasing side entirely
            if position > 0:
                bid_size = 0
                ask_size = min(ask_size, abs_pos)
            else:
                ask_size = 0
                bid_size = min(bid_size, abs_pos)
        elif room < base_size:
            # APPROACHING CAP: reduce increasing-side size to remaining room
            if position > 0:
                bid_size = min(bid_size, room)  # limit buys to remaining room
            elif position < 0:
                ask_size = min(ask_size, room)  # limit sells to remaining room
        # Also: graduated size reduction as position grows toward cap
        # At 70% of cap, start reducing increasing-side size linearly
        if abs_pos > per_ticker_max * 0.7 and abs_pos < per_ticker_max:
            pct_used = abs_pos / per_ticker_max
            scale = max(0.2, 1.0 - (pct_used - 0.7) / 0.3)  # 1.0 at 70%, 0.2 at 100%
            if position > 0:
                bid_size = max(100, int(bid_size * scale))
            elif position < 0:
                ask_size = max(100, int(ask_size * scale))

    # --- Aggregate limit guardrails (CRITICAL for avoiding $10/share fines) ---
    bid_px_out: Optional[float] = bid_px
    ask_px_out: Optional[float] = ask_px

    if agg_position_ratio >= 1.0:
        # OVER LIMIT: block ALL quoting, no exceptions
        return None, None, 0, 0

    if agg_position_ratio >= params.limit_hard_pct:
        if position > 0:
            # Long: only allow selling (ask), block buying (bid)
            bid_px_out = None
            bid_size = 0
            # Cap ask size to current position so we can't overshoot into short territory
            ask_size = min(ask_size, abs(position))
        elif position < 0:
            # Short: only allow buying (bid), block selling (ask)
            ask_px_out = None
            ask_size = 0
            # Cap bid size to current position so we can't overshoot into long territory
            bid_size = min(bid_size, abs(position))
        else:
            # Position == 0: block BOTH sides — any fill increases aggregate
            return None, None, 0, 0

    # Sanity: bid must be < ask
    if bid_px_out is not None and ask_px_out is not None and bid_px_out >= ask_px_out:
        mid_round = round(fair_value, 2)
        bid_px_out = mid_round - 0.01
        ask_px_out = mid_round + 0.01

    return bid_px_out, ask_px_out, bid_size, ask_size


# ---------------------------------------------------------------------------
#  ORDER LIFECYCLE MANAGEMENT
# ---------------------------------------------------------------------------

def cancel_working_order(api: RITClient, order_id: Optional[int]) -> bool:
    """Cancel a single order by ID. Returns True if cancelled or already gone."""
    if order_id is None:
        return True
    result = api.delete_order(order_id)
    return result is not None


def place_and_track(
    api: RITClient,
    ticker: str,
    side: str,
    price: float,
    qty: int,
    ts: TickerState,
) -> Optional[int]:
    """Place a limit order and return its order_id (or None on failure)."""
    result = api.post_order(
        ticker=ticker,
        side=side,
        qty=qty,
        order_type="LIMIT",
        price=price,
        wait=True,
    )
    ts.num_posts += 1
    if result and "order_id" in result:
        return int(result["order_id"])
    return None


def manage_quotes_for_ticker(
    api: RITClient,
    ts: TickerState,
    target_bid_px: Optional[float],
    target_ask_px: Optional[float],
    bid_size: int,
    ask_size: int,
    params: QuotingParams,
    agg_over_limit: bool = False,
) -> None:
    """
    Cancel-and-replace logic for one ticker.
    Only cancels if price has drifted beyond requote_threshold.
    If agg_over_limit is True, only cancel — never place new orders.
    """
    wr = ts.working

    # If aggregate position is over limit, cancel everything and return
    if agg_over_limit:
        if wr.bid_id is not None:
            cancel_working_order(api, wr.bid_id)
            wr.bid_id = None; wr.bid_px = None; wr.bid_qty = 0
        if wr.ask_id is not None:
            cancel_working_order(api, wr.ask_id)
            wr.ask_id = None; wr.ask_px = None; wr.ask_qty = 0
        return

    # --- BID side ---
    if target_bid_px is None:
        # Should NOT be quoting bid → cancel if working
        if wr.bid_id is not None:
            cancel_working_order(api, wr.bid_id)
            ts.num_cancels += 1
            wr.bid_id = None
            wr.bid_px = None
            wr.bid_qty = 0
    else:
        need_new_bid = False
        if wr.bid_id is None:
            need_new_bid = True
        elif wr.bid_px is not None and abs(wr.bid_px - target_bid_px) >= params.requote_threshold:
            cancel_working_order(api, wr.bid_id)
            ts.num_cancels += 1
            wr.bid_id = None
            need_new_bid = True

        if need_new_bid and bid_size > 0:
            oid = place_and_track(api, ts.ticker, "BUY", target_bid_px, bid_size, ts)
            wr.bid_id = oid
            wr.bid_px = target_bid_px
            wr.bid_qty = bid_size

    # --- ASK side ---
    if target_ask_px is None:
        if wr.ask_id is not None:
            cancel_working_order(api, wr.ask_id)
            ts.num_cancels += 1
            wr.ask_id = None
            wr.ask_px = None
            wr.ask_qty = 0
    else:
        need_new_ask = False
        if wr.ask_id is None:
            need_new_ask = True
        elif wr.ask_px is not None and abs(wr.ask_px - target_ask_px) >= params.requote_threshold:
            cancel_working_order(api, wr.ask_id)
            ts.num_cancels += 1
            wr.ask_id = None
            need_new_ask = True

        if need_new_ask and ask_size > 0:
            oid = place_and_track(api, ts.ticker, "SELL", target_ask_px, ask_size, ts)
            wr.ask_id = oid
            wr.ask_px = target_ask_px
            wr.ask_qty = ask_size


# ---------------------------------------------------------------------------
#  POSITION / LIMIT MONITORING
# ---------------------------------------------------------------------------

def sync_positions(api: RITClient, state: BotState) -> None:
    """Read /securities and update TickerState.position for every tracked ticker."""
    secs = api.get_securities()
    if not secs:
        return
    for sec in secs:
        tk = sec.get("ticker", "")
        if tk in state.per_ticker:
            state.per_ticker[tk].position = int(sec.get("position", 0))


def get_aggregate_position(state: BotState) -> int:
    """Sum of |position| across all tracked tickers."""
    return sum(abs(ts.position) for ts in state.per_ticker.values())


def get_limits_info(api: RITClient) -> Tuple[int, int, float, float]:
    """
    Returns (gross_position, gross_limit, gross_fine, net_limit).
    Falls back to safe defaults if the API call fails.
    """
    limits = api.get_limits()
    if not limits:
        return 0, 25000, 5.0, 25000  # safe defaults
    # Limits is a list of limit objects
    lim = limits[0] if limits else {}
    gross = int(lim.get("gross", 0))
    gross_limit = int(lim.get("gross_limit", 25000))
    gross_fine = float(lim.get("gross_fine", 5.0))
    net_limit = int(lim.get("net_limit", 25000))
    return gross, gross_limit, gross_fine, net_limit


# ---------------------------------------------------------------------------
#  FILL DETECTION & STALE ORDER CLEANUP
# ---------------------------------------------------------------------------

def check_and_clear_filled_orders(api: RITClient, state: BotState) -> None:
    """
    Query open orders and reconcile with working OrderRefs.
    If a working order_id is no longer OPEN, it was filled/cancelled — clear it
    so we post a fresh quote on the next cycle.
    
    We also check TRANSACTED orders to distinguish fills from cancels.
    Only fills should increment fill_count_window (for fill-rate tracking).
    """
    open_orders = api.get_orders(status="OPEN")
    if open_orders is None:
        return  # API failure, don't touch anything

    open_ids = {int(o["order_id"]) for o in open_orders if "order_id" in o}

    # Collect IDs that disappeared from OPEN
    disappeared_ids: List[int] = []
    for ts in state.per_ticker.values():
        wr = ts.working
        if wr.bid_id is not None and wr.bid_id not in open_ids:
            disappeared_ids.append(wr.bid_id)
        if wr.ask_id is not None and wr.ask_id not in open_ids:
            disappeared_ids.append(wr.ask_id)

    # Check which disappeared IDs were actually filled (TRANSACTED)
    filled_ids: set = set()
    if disappeared_ids:
        transacted = api.get_orders(status="TRANSACTED")
        if transacted:
            filled_ids = {int(o["order_id"]) for o in transacted if "order_id" in o}

    # Now clear working refs, only counting actual fills for fill-rate
    for ts in state.per_ticker.values():
        wr = ts.working
        if wr.bid_id is not None and wr.bid_id not in open_ids:
            ts.num_fills += 1
            if wr.bid_id in filled_ids:
                ts.fill_count_window += 1  # only real fills count for fill-rate
            wr.bid_id = None
            wr.bid_px = None
            wr.bid_qty = 0
        if wr.ask_id is not None and wr.ask_id not in open_ids:
            ts.num_fills += 1
            if wr.ask_id in filled_ids:
                ts.fill_count_window += 1  # only real fills count for fill-rate
            wr.ask_id = None
            wr.ask_px = None
            wr.ask_qty = 0


# ---------------------------------------------------------------------------
#  MARKET-CLOSE FLATTENING
# ---------------------------------------------------------------------------

def compute_ticks_until_close(current_tick: int, ticks_per_day: int) -> int:
    """
    Returns how many ticks remain until the next "market close".
    Market closes happen every ticks_per_day ticks (default 60).
    """
    if ticks_per_day <= 0:
        return 999
    tick_in_day = current_tick % ticks_per_day
    return ticks_per_day - tick_in_day


def flatten_positions(
    api: RITClient,
    state: BotState,
    ticks_until_close: int,
    params: QuotingParams,
    aggregate_limit: int,
) -> None:
    """
    Reduce ALL positions toward ZERO as market close approaches.
    
    News drops at market open cause regime changes ($1+ price moves).
    Holding ANY inventory through close → open is a losing proposition.
    The goal is to be FLAT (all positions = 0), not just under the limit.
    
    - Soft flatten (flatten_start_ticks_before_close):
      Cancel all working orders, let the quoting section handle reducing.
    - Hard flatten (flatten_hard_ticks_before_close):
      Post passive limit orders to reduce ALL positions toward zero.
    """
    agg_pos = get_aggregate_position(state)

    if ticks_until_close <= params.flatten_hard_ticks_before_close:
        # HARD FLATTEN: reduce ALL positions toward zero, not just excess
        # Any position held through the close boundary is a losing bet
        if agg_pos > 0:
            # Cancel all existing orders first
            api.cancel_all()
            for ts_clean in state.per_ticker.values():
                ts_clean.working = OrderRef()

            # How urgent? Scale price aggressiveness by how close to the boundary
            urgency = 1.0 - (ticks_until_close / max(1, params.flatten_hard_ticks_before_close))
            # urgency: 0.0 at start of hard flatten, 1.0 at the close tick

            print(f"  [FLATTEN-HARD] {ticks_until_close} ticks to close, agg={agg_pos}, urgency={urgency:.1f}")

            sorted_tickers = sorted(
                state.per_ticker.values(),
                key=lambda ts: abs(ts.position),
                reverse=True,
            )
            for ts in sorted_tickers:
                if ts.position == 0:
                    continue

                flatten_qty = min(
                    abs(ts.position),
                    params.flatten_market_order_max,
                )
                side = "SELL" if ts.position > 0 else "BUY"
                mid_px = ts.last_mid or 25.0
                spr = ts.last_spread or 0.10
                # As urgency increases, post closer to mid (more aggressive)
                # At urgency=0: post at 10% of spread (passive)
                # At urgency=1: post at mid (very aggressive, but still limit order)
                offset_frac = 0.10 * (1.0 - urgency)
                if side == "SELL":
                    px = round(mid_px + spr * offset_frac, 2)
                else:
                    px = round(mid_px - spr * offset_frac, 2)
                result = api.post_order(
                    ticker=ts.ticker, side=side, qty=flatten_qty,
                    order_type="LIMIT", price=px, wait=False,
                )
                if result:
                    print(f"    [FLATTEN] {side} {flatten_qty} {ts.ticker} (LIMIT@{px:.2f}, urgency={urgency:.1f})")

    elif ticks_until_close <= params.flatten_start_ticks_before_close:
        # SOFT FLATTEN: cancel all working orders so the quoting section
        # only quotes the reducing side (handled in the main loop)
        if agg_pos > 0:
            print(f"  [FLATTEN-SOFT] {ticks_until_close} ticks to close, agg={agg_pos}/{aggregate_limit}, cancelling all")
            api.cancel_all()
            for ts in state.per_ticker.values():
                ts.working = OrderRef()


# ---------------------------------------------------------------------------
#  TENDER HANDLING
# ---------------------------------------------------------------------------

def handle_tenders(api: RITClient, state: BotState, aggregate_limit: int) -> None:
    """
    DISABLED: Tenders accept massive positions (e.g. 9316 WNTR shares)
    that blow up P&L when the price moves against us.
    The tender at t=242 alone cost ~$7,000.
    Simply decline all tenders.
    """
    tenders = api.get_tenders()
    if not tenders:
        return
    for tender in tenders:
        tid = tender.get("tender_id")
        if tid is not None:
            api.decline_tender(tid)
            qty = abs(int(tender.get("quantity", 0)))
            ticker = tender.get("ticker", "?")
            print(f"  [TENDER] Declined tender {tid}: {qty} shares of {ticker}")


# ---------------------------------------------------------------------------
#  FILL-RATE TRACKING
# ---------------------------------------------------------------------------

def update_fill_rates(state: BotState, params: QuotingParams) -> None:
    """
    Update rolling fill rates for each ticker.
    Called every loop iteration; resets window every fill_rate_window_secs.
    """
    now = time.time()
    for ts in state.per_ticker.values():
        if ts.fill_rate_window_start == 0.0:
            ts.fill_rate_window_start = now

        elapsed = now - ts.fill_rate_window_start
        if elapsed >= params.fill_rate_window_secs:
            # Compute fill rate for this window
            if ts.post_count_window > 0:
                ts.fill_rate = ts.fill_count_window / ts.post_count_window
            else:
                ts.fill_rate = 0.0
            # Reset window
            ts.fill_count_window = 0
            ts.post_count_window = 0
            ts.fill_rate_window_start = now


# ---------------------------------------------------------------------------
#  CROSS-TICKER HEDGING
# ---------------------------------------------------------------------------

def attempt_cross_hedge(
    api: RITClient,
    state: BotState,
    corr: CorrelationTracker,
    agg_limit: int,
    params: QuotingParams,
) -> None:
    """
    DISABLED: Cross-hedging was creating new positions in other tickers via
    market orders, which INCREASED aggregate position instead of reducing it.
    The fundamental flaw: shorting ticker B to "hedge" a long in ticker A
    increases sum(|positions|) unless B is already long and the hedge qty < B's position.
    Keeping the function signature so callers don't break.
    """
    return  # DISABLED — the kill-switch + flatten logic handles limit enforcement


# ---------------------------------------------------------------------------
#  MAIN BOT LOOP
# ---------------------------------------------------------------------------

# Global reference so signal handler can clean up
_global_api: Optional[RITClient] = None


def shutdown_handler(*args):
    """Cancel all orders on exit."""
    if _global_api is not None:
        print("\n[SHUTDOWN] Cancelling all open orders...")
        _global_api.cancel_all()
        print("[SHUTDOWN] Done.")
    sys.exit(0)


def run_market_maker(
    api: RITClient,
    tickers: List[str],
    params: Optional[QuotingParams] = None,
) -> None:
    """
    Main market-making loop.

    1. Wait for case ACTIVE
    2. Sync positions & limits
    3. Detect fills / clean stale orders
    4. Poll news & update fair-value adjustments
    5. Check market-close flattening
    6. For each ticker: refresh book → compute FV (with momentum) → compute quotes (with vol) → manage orders
    7. Handle tenders
    8. Print diagnostics
    9. Sleep & repeat
    """
    global _global_api
    _global_api = api

    if params is None:
        params = QuotingParams()

    state = BotState(tickers=tickers)
    news = NewsTracker(known_tickers=tickers)
    corr = CorrelationTracker(tickers=tickers, window=60)
    momentum = MomentumTracker(tickers=tickers)
    vol = VolatilityTracker(tickers=tickers, window=30)

    # Speed improvement #1: ThreadPoolExecutor for parallel book fetches
    book_executor = ThreadPoolExecutor(max_workers=len(tickers), thread_name_prefix="book")

    last_diag_print = 0.0
    last_news_decay_tick = -1
    last_fill_check_ts = 0.0
    loop_count = 0

    # Speed improvement #3: cache /case polling
    cached_case: Optional[dict] = None
    cached_case_ts: float = 0.0
    case_cache_ttl: float = 0.20  # only re-fetch /case every 200ms

    # Cross-hedge cooldown
    last_hedge_ts: float = 0.0

    # Day boundary tracking — reset momentum/vol when day changes
    last_trading_day: int = -1
    last_tracker_tick: int = -1  # only update trackers once per tick, not every 200ms loop

    # Warmup: reduce size for first N ticks after launch to avoid getting
    # picked off before trackers have data and book is understood
    warmup_ticks_remaining: int = 10
    first_tick_seen: int = -1

    # Performance tracker
    tracker = PerformanceTracker(api, tickers, snapshot_interval=1.0)

    print(f"[BOT] Market maker starting for {tickers}")
    print(f"[BOT] Params: half_spread={params.half_spread}, order_size={params.order_size}, "
          f"inventory_skew={params.inventory_skew_per_share}, "
          f"momentum_fv_shift={params.momentum_fv_shift}, "
          f"vol_baseline={params.vol_baseline}")
    print(f"[BOT] Speed: parallel book fetches, 200ms loop, case caching, book limit=1")
    print(f"[BOT] Strategy: adaptive spread, low-momentum, passive-only, "
          f"no cross-hedge, no rebate-sizing, fill-rate widen-only")

    while True:
        loop_count += 1

        # ---- 1. Case status (cached — Speed improvement #3) ----
        now = time.time()
        if cached_case is None or (now - cached_case_ts) >= case_cache_ttl:
            case = api.get_case()
            if case:
                cached_case = case
                cached_case_ts = now
            else:
                time.sleep(0.1)
                continue
        case = cached_case

        status = case.get("status", "")
        if status != "ACTIVE":
            if status == "STOPPED":
                print("[BOT] Case STOPPED. Exiting loop.")
                break
            cached_case = None  # force re-fetch on next iteration
            time.sleep(0.2)
            continue

        current_tick = int(case.get("tick", 0))
        ticks_per_period = int(case.get("ticks_per_period", 60))
        current_period = int(case.get("period", 1))

        # ---- Warmup tracking ----
        if first_tick_seen < 0:
            first_tick_seen = current_tick
            warmup_ticks_remaining = 10
            print(f"[BOT] First tick seen: {current_tick}, warming up for {warmup_ticks_remaining} ticks")
        else:
            elapsed_ticks = current_tick - first_tick_seen
            warmup_ticks_remaining = max(0, 10 - elapsed_ticks)

        # ---- 2. Sync positions & limits ----
        sync_positions(api, state)
        gross, gross_limit, gross_fine, net_limit = get_limits_info(api)
        agg_pos = get_aggregate_position(state)
        net_position = sum(ts.position for ts in state.per_ticker.values())  # signed sum

        # Update aggregate limit from news if parsed
        if news.parsed_aggregate_limit is not None and not state.aggregate_limit_parsed:
            state.aggregate_limit = news.parsed_aggregate_limit
            state.aggregate_limit_parsed = True
            print(f"[BOT] Aggregate position limit set to {state.aggregate_limit:,} (from news)")

        # Use the MORE CONSERVATIVE of aggregate limit (from news) and gross limit (from API)
        # Both carry fines; aggregate is $10/share (assessed at each market close), gross is $5/share (real-time)
        agg_limit = min(state.aggregate_limit, gross_limit) if gross_limit > 0 else state.aggregate_limit
        if agg_limit != state.aggregate_limit and loop_count % 100 == 0:
            print(f"[BOT] Using effective limit={agg_limit} (min of agg={state.aggregate_limit}, gross={gross_limit})")
        agg_ratio = agg_pos / max(agg_limit, 1)

        # Compute ticks-to-close early (needed by guard + flatten)
        ticks_to_close = compute_ticks_until_close(current_tick, params.ticks_per_day)

        # ---- POSITION GUARD: If aggregate position exceeds limit ----
        if agg_pos > agg_limit:
            # Cancel ALL orders first to stop bleeding
            api.cancel_all()
            for ts_ks in state.per_ticker.values():
                ts_ks.working = OrderRef()

            # Post aggressive LIMIT orders (not market!) to reduce the largest positions.
            # Limit orders earn rebates and don't cross the spread.
            # Only use market orders in the last 3 ticks before close.
            excess = agg_pos - agg_limit
            sorted_ts = sorted(
                state.per_ticker.values(),
                key=lambda ts_sort: abs(ts_sort.position),
                reverse=True,
            )
            # ALWAYS passive limit orders — NEVER market orders.
            for ts_reduce in sorted_ts:
                if excess <= 0:
                    break
                if ts_reduce.position == 0:
                    continue
                reduce_qty = min(abs(ts_reduce.position), excess, params.flatten_market_order_max)
                side = "SELL" if ts_reduce.position > 0 else "BUY"
                # Post at the PASSIVE side of the book to earn rebate
                mid_px = ts_reduce.last_mid or 25.0
                spr = ts_reduce.last_spread or 0.10
                if side == "SELL":
                    px = round(mid_px + spr * 0.05, 2)
                else:
                    px = round(mid_px - spr * 0.05, 2)
                result = api.post_order(
                    ticker=ts_reduce.ticker, side=side, qty=reduce_qty,
                    order_type="LIMIT", price=px, wait=False,
                )
                if result:
                    excess -= reduce_qty
                    print(f"  [GUARD-REDUCE] {side} {reduce_qty} {ts_reduce.ticker} "
                          f"(LIMIT@{px:.2f}, agg={agg_pos}/{agg_limit})")
            # Re-sync after reducing
            sync_positions(api, state)
            agg_pos = get_aggregate_position(state)
            agg_ratio = agg_pos / max(agg_limit, 1)

        # ---- 3. Fill detection (every 150ms) ----
        now = time.time()
        if (now - last_fill_check_ts) >= 0.15:
            check_and_clear_filled_orders(api, state)
            last_fill_check_ts = now

        # ---- 3b. Fill-rate tracking ----
        update_fill_rates(state, params)

        # ---- 4. News polling ----
        news.poll(api)
        if current_tick != last_news_decay_tick:
            news.decay_adjustments()
            last_news_decay_tick = current_tick

        # ---- 5. Market-close flattening ----
        # ticks_to_close already computed above (before guard section)
        ticks_into_day = current_tick % params.ticks_per_day  # ticks since last open
        current_trading_day = current_tick // params.ticks_per_day

        # ---- DAY BOUNDARY RESET ----
        # When a new trading day starts:
        # 1. Cancel ALL orders immediately (don't be in the book during the regime change)
        # 2. Reset momentum & vol trackers (old day's signals are poison)
        # 3. Start a BLACKOUT window — no quoting until the new regime settles
        if current_trading_day != last_trading_day:
            if last_trading_day >= 0:  # don't cancel on first iteration
                print(f"  [DAY-RESET] New trading day {current_trading_day} (tick {current_tick})")
                print(f"  [BLACKOUT] Cancelling all orders, observing price for {params.news_blackout_ticks} ticks")
                api.cancel_all()
                for ts_reset in state.per_ticker.values():
                    ts_reset.working = OrderRef()
                # Log pre-news prices for reference
                for t in tickers:
                    ts_log = state.s(t)
                    if ts_log.last_mid:
                        print(f"    {t}: pre-news mid = {ts_log.last_mid:.2f}")
            momentum.reset()
            vol.reset()
            last_trading_day = current_trading_day

        # ---- NEWS BLACKOUT WINDOW ----
        # During the first N ticks of each trading day, DON'T QUOTE.
        # Just observe the book, let trackers warm up, see where the new regime settles.
        in_blackout = ticks_into_day < params.news_blackout_ticks
        if in_blackout:
            # Still fetch books and update trackers (we want data), but don't post orders
            tickers_to_observe = [t for t in tickers if state.s(t).should_refresh(current_tick)]
            obs_futures = {
                book_executor.submit(api.get_book, ticker=t, limit=1): t
                for t in tickers_to_observe
            }
            for future in as_completed(obs_futures):
                t = obs_futures[future]
                try:
                    book = future.result()
                except Exception:
                    book = None
                ts = state.s(t)
                tob = api.top_of_book(book)
                if tob:
                    mid, spread = api.mid_and_spread(tob)
                    ts.last_best_bid = tob[0]
                    ts.last_best_ask = tob[2]
                    ts.last_mid = mid
                    ts.last_spread = spread
                    ts.mark_refreshed(current_tick)
                    # Update trackers with the new regime's prices
                    if current_tick != last_tracker_tick:
                        corr.update_mid(t, mid)
                        momentum.update(t, mid)
                        vol.update(t, mid)
                        ts.momentum = momentum.get_momentum(t)
                        ts.volatility = vol.get_volatility(t)

            # Log what we see during blackout (once per tick)
            if current_tick != last_tracker_tick and ticks_into_day > 0:
                parts = []
                for t in tickers:
                    ts = state.s(t)
                    if ts.last_mid:
                        parts.append(f"{t}={ts.last_mid:.2f}")
                print(f"  [BLACKOUT] tick {ticks_into_day}/{params.news_blackout_ticks}: {' '.join(parts)}")
            last_tracker_tick = current_tick

            # Handle tenders during blackout (still decline them)
            handle_tenders(api, state, agg_limit)
            time.sleep(0.20)
            continue  # SKIP all quoting logic — stay dark

        is_flattening = ticks_to_close <= params.flatten_start_ticks_before_close

        if is_flattening:
            flatten_positions(api, state, ticks_to_close, params, agg_limit)

        # ---- 5b. Cross-ticker hedging (every 2 seconds, only when not flattening) ----
        now = time.time()
        if not is_flattening and (now - last_hedge_ts) >= 2.0:
            attempt_cross_hedge(api, state, corr, agg_limit, params)
            last_hedge_ts = now

        # ---- 6. Per-ticker quoting (PARALLEL book fetches — Speed improvement #1) ----
        # Determine which tickers need refresh
        tickers_to_refresh = [t for t in tickers if state.s(t).should_refresh(current_tick)]

        # Speed improvement #1 & #2: fetch all books in parallel, limit=1 (only need top-of-book)
        book_futures = {
            book_executor.submit(api.get_book, ticker=t, limit=1): t
            for t in tickers_to_refresh
        }
        book_results: Dict[str, Optional[dict]] = {}
        for future in as_completed(book_futures):
            t = book_futures[future]
            try:
                book_results[t] = future.result()
            except Exception:
                book_results[t] = None

        # Process each ticker's book result
        # Compute aggregate position once (already synced above)
        agg_pos = get_aggregate_position(state)
        agg_ratio = agg_pos / max(agg_limit, 1)
        agg_over = agg_pos > agg_limit

        for t in tickers_to_refresh:
            ts = state.s(t)
            book = book_results.get(t)
            tob = api.top_of_book(book)
            if not tob:
                manage_quotes_for_ticker(api, ts, None, None, 0, 0, params, agg_over_limit=agg_over)
                ts.mark_refreshed(current_tick)
                continue

            mid, spread = api.mid_and_spread(tob)
            ts.last_best_bid = tob[0]
            ts.last_best_ask = tob[2]
            ts.last_mid = mid
            ts.last_spread = spread
            ts.mark_refreshed(current_tick)

            # Update trackers — ONLY when tick has changed.
            # Updating every 200ms loop with the same mid produces zero returns
            # that dilute the signal, and the one big regime-change return
            # stays as an outlier in the vol window for too long.
            tick_changed = (current_tick != last_tracker_tick)
            if tick_changed:
                corr.update_mid(t, mid)
                mom_signal = momentum.update(t, mid)
                vol_ratio_val = vol.update(t, mid)
                ts.momentum = mom_signal
                ts.volatility = vol.get_volatility(t)
            else:
                mom_signal = momentum.get_momentum(t)
                vol_ratio_val = vol.get_volatility(t)

            # Compute fair value (momentum-aware)
            news_adj = news.get_adjustment(t)
            fv = compute_fair_value(mid, news_adj, mom_signal, params)

            # During soft-flatten, only quote the side that reduces inventory
            if is_flattening and ticks_to_close > params.flatten_hard_ticks_before_close:
                urgency = 1.0 - (ticks_to_close - params.flatten_hard_ticks_before_close) / max(1, params.flatten_start_ticks_before_close - params.flatten_hard_ticks_before_close)
                # As urgency rises, post closer to mid to get filled faster
                offset = 0.03 * (1.0 - urgency)  # 3 cents at start, 0 at max urgency
                # Use FULL position as flatten size — we want to get flat, not nibble
                flatten_sz = min(abs(ts.position), params.order_size) if abs(ts.position) > 0 else 0

                if ts.position > 0:
                    ask_price = round(fv + offset, 2)
                    bid_px, ask_px, bid_sz, ask_sz = None, ask_price, 0, flatten_sz
                elif ts.position < 0:
                    bid_price = round(fv - offset, 2)
                    bid_px, ask_px, bid_sz, ask_sz = bid_price, None, flatten_sz, 0
                else:
                    bid_px, ask_px, bid_sz, ask_sz = None, None, 0, 0
            else:
                # Normal quoting with all strategy enhancements
                vol_r = vol.get_vol_ratio(t, params.vol_baseline)
                bid_px, ask_px, bid_sz, ask_sz = compute_quotes(
                    fair_value=fv,
                    position=ts.position,
                    params=params,
                    agg_position_ratio=agg_ratio,
                    side_increasing_long=(ts.position > 0),
                    momentum=mom_signal,
                    vol_ratio=vol_r,
                    ticker=t,
                    market_spread=spread,
                    net_position=net_position,
                    net_limit=net_limit,
                    ticks_into_day=ticks_into_day,
                    fill_rate=ts.fill_rate,
                    agg_limit=agg_limit,
                )

                # End-of-day size tapering: reduce order size as we approach close
                # to avoid building inventory that will need to be flattened.
                # In the last 20 ticks, linearly reduce size (aligns with flatten start).
                if ticks_to_close <= 20:
                    taper = ticks_to_close / 20.0  # 1.0 at 20 ticks, ~0.0 near close
                    taper = max(0.2, taper)  # floor at 20% of normal size
                    if bid_sz > 0:
                        bid_sz = max(100, int(bid_sz * taper))
                    if ask_sz > 0:
                        ask_sz = max(100, int(ask_sz * taper))

                # Warmup size reduction: use small size until trackers have data
                if warmup_ticks_remaining > 0:
                    warmup_scale = 0.25  # 25% of normal size during warmup
                    if bid_sz > 0:
                        bid_sz = max(100, int(bid_sz * warmup_scale))
                    if ask_sz > 0:
                        ask_sz = max(100, int(ask_sz * warmup_scale))

            manage_quotes_for_ticker(api, ts, bid_px, ask_px, bid_sz, ask_sz, params, agg_over_limit=agg_over)

            # Track posts for fill-rate calculation
            if bid_px is not None:
                ts.post_count_window += 1
            if ask_px is not None:
                ts.post_count_window += 1

        # Mark this tick as processed for tracker gating
        last_tracker_tick = current_tick

        # ---- 7. Tenders ----
        handle_tenders(api, state, agg_limit)

        # ---- 8. Diagnostics & performance tracking (every 3 seconds) ----
        now = time.time()
        if (now - last_diag_print) > 3.0:
            # Record snapshot (writes CSV row, returns TickSnapshot or None)
            snap = tracker.record(current_tick, current_period, state, params, news, vol, momentum, corr)

            # Print live dashboard (compact P&L + per-ticker breakdown)
            if snap:
                tracker.print_live_dashboard(snap)

            # Also print working-order details and correlation
            ttc_str = f"close_in={ticks_to_close}" if ticks_to_close <= params.flatten_start_ticks_before_close + 5 else ""
            net_str = f"net={net_position:+d}/{net_limit}"
            print(f"  [orders] agg_pos={agg_pos}/{agg_limit} ({agg_ratio:.0%}) gross={gross}/{gross_limit} {net_str} {ttc_str}")
            for t in tickers:
                ts = state.s(t)
                wr = ts.working
                print(
                    f"    {t}: bid={wr.bid_px or 0:.2f}x{wr.bid_qty}  "
                    f"ask={wr.ask_px or 0:.2f}x{wr.ask_qty}  "
                    f"fills={ts.num_fills} posts={ts.num_posts}"
                )

            # Print correlation matrix periodically
            if corr.ready(min_points=20):
                mat = corr.corr_matrix()
                pairs = []
                for i in range(len(tickers)):
                    for j in range(i + 1, len(tickers)):
                        a, b = tickers[i], tickers[j]
                        pairs.append(f"{a}-{b}:{mat[(a,b)]:+.2f}")
                print(f"  corr: {' | '.join(pairs)}")

            last_diag_print = now

        # ---- 9. Pacing ----
        time.sleep(0.20)

    # Clean up on normal exit
    print("[BOT] Saving performance data...")
    tracker.save()
    tracker.print_summary()
    print("[BOT] Cancelling all orders before exit...")
    api.cancel_all()
    book_executor.shutdown(wait=False)
    print("[BOT] Done.")


# ---------------------------------------------------------------------------
#  ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api = RITClient()
    tickers = ["SPNG", "SMMR", "ATMN", "WNTR"]

    # Register cleanup handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    atexit.register(lambda: api.cancel_all())

    # Tunable strategy parameters — adjust between heats
    params = QuotingParams(
        # --- Base quoting ---
        half_spread=0.10,             # $0.10 each side — wider fallback for safety
        order_size=2000,              # 2000 shares per side — doubled from 1000 for more rebates
        inventory_skew_per_share=0.0005,  # $0.0005 skew per share — gentle nudge, not sledgehammer
        max_inventory_skew=0.04,      # cap skew at $0.04 — never more than half the spread
        requote_threshold=0.03,       # re-quote if price moves $0.03 (less cancel churn)
        # --- Position limit guardrails ---
        limit_warning_pct=0.70,       # start widening at 70% of aggregate limit (was 60%)
        limit_extra_spread=0.08,      # add up to $0.08 spread near limit (was $0.10)
        limit_size_reduction=0.50,    # reduce size by up to 50% near limit
        limit_hard_pct=0.85,          # stop quoting one side at 85% of limit (was 80%)
        # --- Momentum / adverse selection ---
        momentum_fv_shift=0.5,        # small — stop trend-following
        momentum_spread_add=0.5,      # small — stop over-widening
        momentum_dead_zone=0.001,     # ignore small momentum noise
        # --- Volatility-adaptive spread ---
        vol_spread_enabled=True,
        vol_baseline=0.005,           # baseline vol
        vol_max_multiplier=2.0,       # cap vol multiplier at 2x
        # --- Market-close flattening ---
        ticks_per_day=60,             # 60 ticks = 1 minute = 1 trading day
        flatten_start_ticks_before_close=18,  # soft-flatten 18 ticks early (was 20 — more quoting time)
        flatten_hard_ticks_before_close=8,    # hard flatten 8 ticks early (was 10 — more quoting time)
        flatten_market_order_max=5000,        # max shares per flatten order
        # --- Per-ticker rebate sizing ---
        ticker_rebates={"SPNG": 0.03, "SMMR": 0.04, "ATMN": 0.035, "WNTR": 0.045},
        rebate_sizing_enabled=False,          # DISABLED
        rebate_baseline=0.03,
        # --- Adaptive spread from book ---
        adaptive_spread_enabled=True,         # use actual market spread
        adaptive_spread_fraction=1.0,         # match market spread exactly
        adaptive_spread_floor=0.08,           # NEVER tighter than $0.08 half-spread
        adaptive_spread_ceiling=0.25,         # never wider than $0.25 half-spread
        # --- News blackout window ---
        news_blackout_ticks=7,                # don't quote for first 8 ticks of each day
        # --- Net limit awareness ---
        net_limit_warning_pct=0.50,           # start net-skew at 50% of net limit
        net_skew_per_share=0.0003,
        # --- Per-ticker position cap ---
        per_ticker_limit_fraction=0.30,       # each ticker max 30% of aggregate limit (was 25% — more room)
        # --- Fill-rate auto-tuning ---
        fill_rate_target=0.30,                # target 30% fill rate
        fill_rate_window_secs=30.0,           # measure over 30 second windows
        fill_rate_max_adjust=0.02,            # max $0.02 widen from fill-rate (never tightens)
    )

    print("=" * 60)
    print("  RIT Algorithmic Market Maker")
    print("  Tickers:", tickers)
    print("=" * 60)

    run_market_maker(api, tickers, params)
