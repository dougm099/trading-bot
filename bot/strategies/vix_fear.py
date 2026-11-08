"""VIX fear-spike mean reversion (EXPERIMENTAL).

Long SPY when the VIX term structure inverts: spot VIX above 3-month VIX
(VIX3M/VIX < 1.0), the classic "buy the panic" setup. Holds 20 trading days,
long-only, one position at a time, no pyramiding.

Status: EXPERIMENTAL, thin sample (~15 independent episodes in 5y of data).
Mean +1.7%/trade, 67% hit rate, positive in both split-sample halves, and
costs are negligible (10 bps round-trip vs ~170 bps mean trade), but n=15 is
small and the second half was weaker. Treat as a tactical overlay candidate,
not a standalone strategy. NOT wired to paper/live trading.

Research note: results/outside_the_box.md (VIX term-structure thread).
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..models import Signal
from .base import Strategy


class VixFearReversion(Strategy):
    name = "vix_fear_reversion"
    timeframe = "1d"
    backtestable = True

    def __init__(
        self,
        vix_slope: Optional[pd.Series] = None,
        hold_days: int = 20,
        threshold: float = 1.0,
    ):
        """vix_slope: daily VIX3M/VIX ratio indexed by date (ascending).

        If None, fetched once via yfinance (^VIX, ^VIX3M). Pass explicitly in
        backtests for reproducibility.
        """
        if vix_slope is None:
            vix_slope = _fetch_vix_slope()
        s = vix_slope.copy()
        s.index = pd.to_datetime(s.index).date
        self.slope = s.sort_index()
        self.hold_days = hold_days
        self.threshold = threshold
        self._entry_date = None  # date of current open episode, else None

    def _slope_asof(self, day) -> Optional[float]:
        s = self.slope[self.slope.index <= day]
        return float(s.iloc[-1]) if len(s) else None

    def on_bar(self, symbol: str, bars: pd.DataFrame) -> List[Signal]:
        if len(bars) < 2:
            return []
        day = pd.to_datetime(bars.index[-1]).date()
        slope = self._slope_asof(day)
        if slope is None:
            return []

        if self._entry_date is not None:
            held = (day - self._entry_date).days
            # trading-day approximation: ~21 trading days per 30 calendar days
            if held >= int(self.hold_days * 30 / 21):
                self._entry_date = None
                return [Signal(symbol, "flat", strength=1.0,
                               reason=f"vix fear hold expired ({self.hold_days}d)")]
            return []  # still holding

        if slope < self.threshold:
            self._entry_date = day
            return [Signal(symbol, "buy", strength=1.0,
                           reason=f"VIX curve inverted: VIX3M/VIX={slope:.2f}<{self.threshold}")]
        return []


def _fetch_vix_slope(period: str = "5y") -> pd.Series:
    import yfinance as yf

    raw = yf.download(["^VIX", "^VIX3M"], period=period, interval="1d",
                      progress=False, auto_adjust=True)["Close"]
    df = pd.DataFrame({"VIX": raw["^VIX"], "VIX3M": raw["^VIX3M"]}).dropna()
    return df["VIX3M"] / df["VIX"]
