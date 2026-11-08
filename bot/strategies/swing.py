"""Swing trend-following: EMA fast/slow cross with ATR trend filter.

Holds days to weeks. Long when fast EMA > slow EMA and price is above the
slow EMA by a fraction of ATR (avoids chop); flat/exit on cross down.
"""
from __future__ import annotations

from typing import List

import pandas as pd

from ..models import Signal
from .base import Strategy, atr, ema


class SwingTrend(Strategy):
    name = "swing_trend"

    def __init__(self, fast: int = 12, slow: int = 48, atr_buffer: float = 0.25):
        self.fast = fast
        self.slow = slow
        self.atr_buffer = atr_buffer

    def on_bar(self, symbol: str, bars: pd.DataFrame) -> List[Signal]:
        if len(bars) < self.slow + 5:
            return []
        c = bars["close"]
        f, s = ema(c, self.fast), ema(c, self.slow)
        a = atr(bars).iloc[-1]
        px = c.iloc[-1]
        bull = f.iloc[-1] > s.iloc[-1] and px > s.iloc[-1] + self.atr_buffer * a
        bear = f.iloc[-1] < s.iloc[-1]
        was_bull = f.iloc[-2] > s.iloc[-2]

        if bull and not was_bull:
            return [Signal(symbol, "buy", strength=0.8, reason="EMA cross up")]
        if bear and was_bull:
            return [Signal(symbol, "flat", reason="EMA cross down")]
        return []
