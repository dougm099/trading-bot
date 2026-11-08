"""Triple-barrier labeling (Lopez de Prado, Advances in Financial ML).

For each bar, look ahead up to `horizon` bars:
  +1  if the upper barrier (take-profit) is touched first
  -1  if the lower barrier (stop-loss) is touched first
   0  if neither is touched within the horizon (vertical barrier)

Barriers scale with recent ATR so labels adapt to volatility regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategies.base import atr


def triple_barrier_labels(bars: pd.DataFrame, horizon: int = 12,
                          tp_mult: float = 1.5, sl_mult: float = 1.0) -> pd.Series:
    c, h, l = bars["close"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy()
    a = atr(bars, 14).to_numpy()
    n = len(bars)
    labels = np.zeros(n, dtype=int)

    for i in range(n - 1):
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        entry = c[i]
        tp = entry + tp_mult * a[i]
        sl = entry - sl_mult * a[i]
        end = min(i + horizon, n - 1)
        for j in range(i + 1, end + 1):
            hit_tp = h[j] >= tp
            hit_sl = l[j] <= sl
            if hit_tp and hit_sl:
                labels[i] = 1 if (tp - entry) <= (entry - sl) else -1
                break
            if hit_tp:
                labels[i] = 1
                break
            if hit_sl:
                labels[i] = -1
                break
    return pd.Series(labels, index=bars.index, name="label")
