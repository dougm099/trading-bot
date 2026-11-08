"""Point-in-time feature builder for the long-horizon ML strategy (live use).

Replicates the exact feature definitions from
scripts/ml_longterm_experiment.py::build_samples, but for a single scoring
date (the latest close) and without labels. Output is the RAW feature frame
that bot/strategies/longterm_ml.score_universe() expects (it z-scores
internally).

Features (all point-in-time as of the last bar):
  mom_12_1, mom_6, rev_1m, dist_52w_high, vol_63, dd_252, turnover_trend,
  sector_rel_mom, mkt_trend, vix_level
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

FEATURES = ["mom_12_1", "mom_6", "rev_1m", "dist_52w_high", "vol_63", "dd_252",
            "turnover_trend", "sector_rel_mom", "mkt_trend", "vix_level"]
EXCLUDE = ("SPY", "^VIX", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
           "XLY", "XLU", "XLRE", "XLB", "XLC")
MIN_BARS = 400
VIX_WINDOW = 1260  # ~5y of trading days for the VIX z-score


def build_today_features(closes: pd.DataFrame, highs: pd.DataFrame,
                         volumes: pd.DataFrame,
                         sectors: Dict[str, str]) -> pd.DataFrame:
    """One row per ticker, columns == FEATURES, index = ticker.

    closes/highs/volumes: DataFrames indexed by date (ascending), ticker columns.
    sectors: ticker -> sector ETF symbol (for sector_rel_mom).
    Tickers that fail data requirements are dropped (same spirit as the
    research pipeline's filters).
    """
    idx = closes.index
    i = len(idx) - 1
    tickers = [t for t in closes.columns if t not in EXCLUDE]

    spy = closes["SPY"].to_numpy()
    vix_raw = closes["^VIX"].to_numpy()
    mkt_trend = float(spy[i] / spy[i - 63] - 1)
    vw = vix_raw[max(0, i - VIX_WINDOW):i + 1]
    vw = vw[~np.isnan(vw)]
    vix_level = float((vix_raw[i] - vw.mean()) / vw.std()) \
        if len(vw) > 100 and np.std(vw) > 1e-9 else np.nan

    # sector ETF 12-1 momentum as of today
    sec_mom = {}
    for etf in sorted(set(sectors.values())):
        if etf not in closes.columns:
            continue
        e = closes[etf].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            sec_mom[etf] = float(e[i - 21] / e[i - 252] - 1)

    rows, mom12 = {}, {}
    for t in tickers:
        c = closes[t].dropna()
        if len(c) < MIN_BARS or i - 252 < 0:
            continue
        cv = c.to_numpy()
        # align highs/volumes to the ticker's own valid index
        hv = highs[t].reindex(c.index).to_numpy()
        vv = volumes[t].reindex(c.index).to_numpy()
        j = len(c) - 1  # position of latest bar in ticker-local arrays
        if j - 252 < 0:
            continue
        hmax = float(np.nanmax(hv[j - 251:j + 1]))
        lr = np.log(cv[j - 62:j + 1])
        vol63 = float(np.nanstd(lr) * math.sqrt(252))
        vals = [cv[j], cv[j - 21], cv[j - 252], cv[j - 126], cv[j - 63],
                hmax, vol63]
        if not np.all(np.isfinite(vals)):
            continue
        if cv[j - 252] == 0 or hmax == 0:
            continue
        v21 = float(np.nanmean(vv[j - 20:j + 1]))
        v63 = float(np.nanmean(vv[j - 62:j + 1]))
        if not np.isfinite(v21) or not np.isfinite(v63) or v63 <= 0:
            continue
        w = cv[j - 252:j + 1]
        dd = float(np.min(w / np.maximum.accumulate(w) - 1))
        m12 = float(cv[j - 21] / cv[j - 252] - 1)
        etf = sectors.get(t, "")
        smv = sec_mom.get(etf, np.nan)
        srel = m12 - smv if np.isfinite(smv) else np.nan
        rows[t] = [m12, float(cv[j] / cv[j - 126] - 1),
                   float(cv[j] / cv[j - 21] - 1), float(cv[j] / hmax - 1),
                   vol63, dd, float(v21 / v63), srel, mkt_trend, vix_level]
        mom12[t] = m12

    df = pd.DataFrame.from_dict(rows, orient="index", columns=FEATURES)
    # sector fallback: universe-mean-demeaned momentum (mirrors research)
    umean = float(np.mean(list(mom12.values()))) if mom12 else np.nan
    mask = df["sector_rel_mom"].isna()
    df.loc[mask, "sector_rel_mom"] = df.loc[mask, "mom_12_1"] - umean
    return df.dropna(subset=FEATURES)
