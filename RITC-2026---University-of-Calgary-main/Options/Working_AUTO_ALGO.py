"""
volatility_algo.py
------------------

This module implements a simple trading algorithm inspired by the
2026 Rotman International Trading Competition's Volatility Trading Case.

The case describes a market in which participants receive periodic news
updates about the *realized* (historical) volatility of an ETF (called RTM)
as well as forecasts of the next week's volatility. Market makers quote
option prices using the Black–Scholes model without a variance risk
premium, so implied volatility will converge to the realized volatility
over time. Informed analysts, however, have better estimates of future
volatility than the market maker. When a new volatility shock occurs, a
window of mispricing may arise until the market maker updates their
forecasts. Traders can profit by taking positions in options and
hedging the delta exposure of the underlying ETF.

This algorithm does **not** attempt to place actual trades; instead, it
demonstrates how one might parse news releases to detect changes in
expected volatility and decide whether to take a long or short volatility
position. The logic is simplistic: if the forecasted volatility is higher
than the previous estimate, the algorithm chooses a long volatility
strategy (e.g. buy straddles or strangles); if the forecasted volatility
is lower, it opts for a short volatility strategy (e.g. sell options).

Usage Example:

    >>> news_feed = [
    ...     "The analysts have informed you that the realized volatility of RTM for this week will be 18%",  
    ...     "The analysts have informed you that the realized volatility of RTM next week will be between 28% and 33%",  
    ...     "The analysts have informed you that the realized volatility of RTM for this week will be 29%"
    ... ]
    >>> algo = VolatilityTradingAlgo()
    >>> for news in news_feed:
    ...     decision = algo.process_news(news)
    ...     if decision:
    ...         print(decision)

This will output something like:

    {'news': 'The analysts have informed you that the realized volatility of RTM next week will be between 28% and 33%',
     'expected_vol': 30.5,
     'direction': 'long_volatility',
     'explanation': 'Forecast increased from 18.0% to 30.5%, suggesting long volatility strategy.'}

Note: In a real trading system, decisions would trigger actual trades and
incorporate more sophisticated risk management, delta hedging, and
consideration of transaction costs and limits. This example is meant
solely as a demonstration of parsing news and mapping expected volatility
changes to a directional view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class VolatilityTradingAlgo:
    """A simple algorithm that processes news headlines about volatility.

    Attributes
    ----------
    last_expected_vol : Optional[float]
        The most recently processed expected volatility (in percentage
        points). It is updated whenever a news release provides a new
        forecast. When no previous estimate exists, it remains ``None``.

    history : List[Tuple[str, float]]
        A chronological log of processed news items and the associated
        expected volatility extracted from each.
    """

    last_expected_vol: Optional[float] = None
    history: List[Tuple[str, float]] = field(default_factory=list)

    # Regular expression to match volatility percentages or ranges in news.
    VOL_PATTERN = re.compile(
        r"(?P<low>\d+(?:\.\d+)?)%\s*(?:and|to|–|-)\s*(?P<high>\d+(?:\.\d+)?)%"
        r"|(?P<single>\d+(?:\.\d+)?)%"
    )

    def _parse_volatility(self, news: str) -> Optional[float]:
        """Extract an expected volatility value from a news string.

        The news releases in the RITC volatility case often take forms such
        as:

        - "The analysts have informed you that the realized volatility of RTM for this week will be 18%"
        - "The analysts have informed you that the realized volatility of RTM next week will be between 28% and 33%"

        This helper searches for a percentage or range of percentages in the
        string. If a range is provided (e.g. "28% and 33%"), it returns the
        midpoint of the range. If a single percentage is provided, it returns
        that value directly. If no percentage can be found, it returns None.

        Parameters
        ----------
        news : str
            A news headline or message containing volatility information.

        Returns
        -------
        Optional[float]
            The extracted expected volatility (e.g. 30.5 for a range of
            28%–33%). Returns ``None`` if parsing fails.
        """
        match = self.VOL_PATTERN.search(news)
        if not match:
            return None
        if match.group('low') and match.group('high'):
            low = float(match.group('low'))
            high = float(match.group('high'))
            return (low + high) / 2.0
        elif match.group('single'):
            return float(match.group('single'))
        return None

    def process_news(self, news: str) -> Optional[Dict[str, object]]:
        """Process a single news item and decide on a volatility trading direction.

        This method attempts to extract an expected volatility from the news
        message. If successful, it compares the new value with the last
        stored expected volatility. When the new expected volatility is higher
        than the previous one, it suggests a **long volatility** position;
        when lower, it suggests a **short volatility** position. If this is
        the first news item processed (no prior volatility), it simply
        records the value and returns ``None`` because a directional
        comparison cannot yet be made.

        Parameters
        ----------
        news : str
            The news item describing expected or realized volatility.

        Returns
        -------
        Optional[Dict[str, object]]
            A dictionary containing the news, extracted expected volatility,
            the suggested direction ('long_volatility' or 'short_volatility'),
            and a human-readable explanation. Returns ``None`` if no action
            should be taken (e.g. the first news item or parsing failure).
        """
        expected_vol = self._parse_volatility(news)
        if expected_vol is None:
            # No volatility information found; ignore this news.
            return None

        decision: Optional[Dict[str, object]] = None
        if self.last_expected_vol is not None:
            # Compare with previous expected volatility to determine direction.
            if expected_vol > self.last_expected_vol:
                direction = 'long_volatility'
                explanation = (
                    f"Forecast increased from {self.last_expected_vol:.1f}% to {expected_vol:.1f}%, "
                    f"suggesting long volatility strategy."
                )
            elif expected_vol < self.last_expected_vol:
                direction = 'short_volatility'
                explanation = (
                    f"Forecast decreased from {self.last_expected_vol:.1f}% to {expected_vol:.1f}%, "
                    f"suggesting short volatility strategy."
                )
            else:
                direction = 'no_change'
                explanation = (
                    f"Forecast unchanged at {expected_vol:.1f}%; no directional bias."
                )
            decision = {
                'news': news,
                'expected_vol': expected_vol,
                'direction': direction,
                'explanation': explanation,
            }

        # Update state and history.
        self.last_expected_vol = expected_vol
        self.history.append((news, expected_vol))
        return decision

    def reset(self) -> None:
        """Reset the algorithm's state.

        Clears the last expected volatility and the history. This can be
        useful to begin processing a new sequence of news without retaining
        previous context.
        """
        self.last_expected_vol = None
        self.history.clear()


if __name__ == '__main__':
    # Example usage when run as a script. This demonstrates how the
    # algorithm processes a series of news headlines.
    sample_news = [
        "The analysts have informed you that the realized volatility of RTM for this week will be 18%",
        "The analysts have informed you that the realized volatility of RTM next week will be between 28% and 33%",
        "The analysts have informed you that the realized volatility of RTM for this week will be 29%",
        "The analysts have informed you that the realized volatility of RTM next week will be between 25% and 27%",
    ]

    algo = VolatilityTradingAlgo()
    for msg in sample_news:
        decision = algo.process_news(msg)
        if decision:
            print(decision)