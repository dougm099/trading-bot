"""Feature engineering: OHLCV bars -> model-ready feature matrix.

Same idea as bioinformatics feature extraction: raw observations in,
numeric descriptors out. Every feature uses only data available at or
before the bar's timestamp (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategies.base import atr, ema, rsi

FEATURES: list[str] = []


def _reg(name: str):
    if name not in FEATURES:
        FEATURES.append(name)
    return name


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: datetime-indexed OHLCV, ascending. Returns feature DataFrame."""
    df = pd.DataFrame(index=bars.index)
    c, h, l, v = bars["close"], bars["high"], bars["low"], bars["volume"]

    # multi-horizon returns
    for n in (1, 3, 6, 12, 24):
        df[_reg(f"ret_{n}")] = c.pct_change(n)

    # momentum / oscillators
    df[_reg("rsi_14")] = rsi(c, 14) / 100.0
    df[_reg("rsi_7")] = rsi(c, 7) / 100.0

    # trend: price vs EMAs
    e12, e48 = ema(c, 12), ema(c, 48)
    df[_reg("close_ema12")] = c / e12 - 1
    df[_reg("close_ema48")] = c / e48 - 1
    df[_reg("ema12_ema48")] = e12 / e48 - 1

    # volatility regime
    a14 = atr(bars, 14)
    df[_reg("atr_pct")] = a14 / c
    df[_reg("vol_12")] = c.pct_change().rolling(12).std()
    df[_reg("vol_48")] = c.pct_change().rolling(48).std()
    df[_reg("vol_ratio")] = df["vol_12"] / (df["vol_48"] + 1e-12)

    # volume
    df[_reg("vol_z")] = v / (v.rolling(24).mean() + 1e-12) - 1

    # range position: where did we close inside the recent range?
    hh = h.rolling(24).max()
    ll = l.rolling(24).min()
    df[_reg("range_pos")] = (c - ll) / (hh - ll + 1e-12)

    # intraday seasonality
    df[_reg("hour")] = bars.index.hour / 23.0
    df[_reg("dow")] = bars.index.dayofweek / 6.0

    # --- v2 features (all lookahead-free: computed from bars at/before t) ---

    # Bollinger band position (20, 2): where is price inside the bands?
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df[_reg("bb_pos")] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-12)
    df[_reg("bb_width")] = (4 * bb_std) / (bb_mid + 1e-12)  # squeeze/expansion

    # MACD histogram normalized by price
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    df[_reg("macd_hist")] = (macd_line - macd_sig) / (c + 1e-12)

    # Stochastic %K (14)
    ll14 = l.rolling(14).min()
    hh14 = h.rolling(14).max()
    df[_reg("stoch_k")] = (c - ll14) / (hh14 - ll14 + 1e-12)

    # Parkinson volatility (14): uses high/low range, more efficient than close-only
    df[_reg("parkinson")] = (np.log(h / l) ** 2).rolling(14).mean().pipe(np.sqrt)

    # inter-bar gap: (open - prior close) / prior close (24/7 crypto: session-less momentum gap)
    df[_reg("gap")] = bars["open"] / (c.shift(1) + 1e-12) - 1

    # distance from rolling VWAP (24 bars)
    typical = (h + l + c) / 3
    vwap = (typical * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-12)
    df[_reg("dist_vwap")] = c / (vwap + 1e-12) - 1

    return df.replace([np.inf, -np.inf], np.nan)
