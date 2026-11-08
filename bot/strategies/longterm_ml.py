"""Long-horizon cross-sectional ML stock selection (research artifact).

Validated in results/ml_longterm.md (walk-forward 2016-2025, OOS):
  primary config hgb_d3 / 63d label: net Sharpe 1.29 vs 1.03 benchmark,
  net CAGR 47.7%, DSR=1.000 (K=4), maxDD -49%.

This module is PURE selection logic: score a month-end feature frame and
return the top-N tickers. It is intentionally NOT a Strategy subclass and is
not referenced by any runner, paper or live. Promotion to execution would
require a fresh decision and forward validation.

Feature contract (point-in-time, computed as of the month-end signal close):
  mom_12_1, mom_6, rev_1m, dist_52w_high, vol_63, dd_252, turnover_trend,
  sector_rel_mom, mkt_trend, vix_level
See scripts/ml_longterm_experiment.py build_samples() for exact definitions.
Cross-sectional z-scoring of the 8 stock-specific features is applied per
date inside score_universe(); mkt_trend/vix_level pass through unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "longterm_ml_hgb_d3_63d.joblib"
FEATURES: List[str] = ["mom_12_1", "mom_6", "rev_1m", "dist_52w_high", "vol_63",
                       "dd_252", "turnover_trend", "sector_rel_mom",
                       "mkt_trend", "vix_level"]
STOCK_FEATURES: List[str] = [f for f in FEATURES if f not in ("mkt_trend", "vix_level")]
TOP_N = 25


def load_bundle(path: str | Path = MODEL_PATH) -> dict:
    bundle = joblib.load(path)
    assert bundle["features"] == FEATURES, "feature contract mismatch"
    return bundle


def score_universe(feature_frame: pd.DataFrame, bundle: dict | None = None) -> pd.Series:
    """Rank the universe for one signal date.

    feature_frame: one row per ticker, columns == FEATURES (raw, point-in-time).
    Returns P(top-tercile 63d forward return), indexed like the input, sorted desc.
    """
    bundle = bundle or load_bundle()
    missing = [f for f in FEATURES if f not in feature_frame.columns]
    if missing:
        raise ValueError(f"missing features: {missing}")
    X = feature_frame[FEATURES].copy()
    mu, sd = X[STOCK_FEATURES].mean(), X[STOCK_FEATURES].std().clip(lower=1e-8)
    X[STOCK_FEATURES] = (X[STOCK_FEATURES] - mu) / sd
    proba = bundle["model"].predict_proba(X[FEATURES].to_numpy())[:, 1]
    return pd.Series(proba, index=feature_frame.index).sort_values(ascending=False)


def select_top_n(feature_frame: pd.DataFrame, n: int = TOP_N,
                 bundle: dict | None = None) -> List[str]:
    """Tickers to hold until the next monthly rebalance (equal weight)."""
    return score_universe(feature_frame, bundle).head(n).index.tolist()


def target_weights(tickers: Sequence[str]) -> pd.Series:
    """Equal-weight target portfolio for the selected names."""
    if not tickers:
        raise ValueError("empty selection")
    return pd.Series(1.0 / len(tickers), index=list(tickers))
