"""Fixed-width window fractional differentiation (FFD).

Lopez de Prado, Advances in Financial Machine Learning, Ch. 5.

Returns (d=1) are stationary but memoryless; prices (d=0) remember but are
non-stationary. FFD finds the *minimum* d that passes an ADF stationarity
test (often 0.3-0.6): the transformed series stays stationary while retaining
~98% correlation with the price level, vs ~1% for returns.

The transform is causal: f_t uses only x_s for s <= t (no lookahead).
Fit d* on TRAIN data only. Never on the full sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def fracdiff_weights(d: float, thres: float = 1e-5, max_w: int = 10000) -> np.ndarray:
    """Weight vector w[k], k=0..W-1, with w[0]=1 multiplying x_t.

    w_k = -w_{k-1} * (d - k + 1) / k ; truncate when |w_k| < thres.
    """
    w = [1.0]
    k = 1
    while k <= max_w:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w)


def fracdiff(x: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """Fixed-width window fractional differentiation of x at order d.

    f_t = sum_{k=0}^{W-1} w_k * x_{t-k}. First W-1 values are NaN (causal).
    """
    w = fracdiff_weights(d, thres)
    W = min(len(w), len(x))  # can't look back further than the series itself
    w = w[:W]
    out = np.full(len(x), np.nan)
    xv = x.to_numpy(dtype=float)
    # vectorized via cumulative weighted shifts
    acc = np.zeros(len(x))
    for k, wk in enumerate(w):
        acc[k:] += wk * xv[: len(x) - k]
    out[W - 1 :] = acc[W - 1 :]
    return pd.Series(out, index=x.index)


def find_min_stationary_d(x: pd.Series, d_grid: np.ndarray | None = None,
                          pval: float = 0.05, thres: float = 1e-5) -> tuple[float, pd.DataFrame]:
    """Smallest d in d_grid whose FFD series passes ADF at level `pval`.

    x: the TRAIN-only series (log prices). Returns (d*, diagnostics table).
    If nothing passes, returns (1.0, table), i.e. fall back to returns.
    """
    if d_grid is None:
        d_grid = np.round(np.arange(0.05, 1.0001, 0.05), 4)
    rows = []
    for d in d_grid:
        f = fracdiff(x, float(d), thres).dropna()
        try:
            stat, pv, *_ = adfuller(f, autolag="AIC")
        except Exception:
            stat, pv = np.nan, np.nan
        rows.append({"d": float(d), "adf_stat": stat, "adf_pval": pv,
                     "stationary": bool(pv < pval)})
    diag = pd.DataFrame(rows)
    ok = diag[diag["stationary"]]
    d_star = float(ok["d"].min()) if len(ok) else 1.0
    return d_star, diag
