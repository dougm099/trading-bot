"""Intraday momentum breakout (day trading).

Long when price breaks above the rolling high with expanding range; short on
breakdown. Exits on opposite break or end-of-signal. Designed for 5m-1h bars;
in backtests it also works on hourly crypto bars held < 24h.
"""
from __future__ import annotations

from typing import List

import pandas as pd

from ..models import Signal
from .base import Strategy, atr


class MomentumBreakout(Strategy):
    name = "momentum_breakout"

    def __init__(self, lookback: int = 20, atr_mult: float = 1.5,
                 min_atr_pct: float = 0.15, max_hold_bars: int = 24):
        self.lookback = lookback
        self.atr_mult = atr_mult
        self.min_atr_pct = min_atr_pct      # skip dead markets
        self.max_hold_bars = max_hold_bars

    def on_bar(self, symbol: str, bars: pd.DataFrame) -> List[Signal]:
        if len(bars) < self.lookback + 5:
            return []
        c = bars["close"]
        a = atr(bars).iloc[-1]
        px = c.iloc[-1]
        if a / px * 100 < self.min_atr_pct:
            return [Signal(symbol, "flat", reason="low volatility regime")]

        hi = bars["high"].rolling(self.lookback).max().iloc[-2]  # prior-bar high
        lo = bars["low"].rolling(self.lookback).min().iloc[-2]
        prev = c.iloc[-2]

        if px > hi and prev <= hi:
            return [Signal(symbol, "buy", strength=min(1.0, (px - hi) / (a + 1e-9)),
                           reason=f"breakout above {hi:.2f}")]
        if px < lo and prev >= lo:
            return [Signal(symbol, "sell", strength=min(1.0, (lo - px) / (a + 1e-9)),
                           reason=f"breakdown below {lo:.2f}")]
        return []
