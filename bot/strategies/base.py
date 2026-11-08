"""Strategy interface: pure functions of bars -> signals. No broker access
except optional read-only chain lookups passed in by the runner."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from ..models import Signal


class Strategy(ABC):
    name: str = "base"
    timeframe: str = "1h"

    @abstractmethod
    def on_bar(self, symbol: str, bars: pd.DataFrame) -> List[Signal]:
        """Given full bar history (ascending, datetime index), emit signals."""


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, 1e-9))
