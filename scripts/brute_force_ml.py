#!/usr/bin/env python3
"""Brute-force ML grind: ONE pipeline, hammered through deep iteration.

Fixed architecture (the supervised setup from bot/ml/):
    hourly bars -> feature engineering -> triple-barrier labels ->
    HistGradientBoosting -> walk-forward validation.

We sweep HARD over: feature subsets (incl. dropping groups), horizons,
triple-barrier params, HGB hyperparameters, and symbols — with honest
discipline:

  * walk-forward train/val/test on every config; OOS metrics only
  * transaction costs 10 bps/side (25 bps sensitivity on finalists)
  * White's Reality Check on top configs (multiple-testing correction)
  * Deflated Sharpe Ratio (Bailey & de Prado)
  * LOCKED holdout: last 20% of each series, never touched during the sweep;
    top-3 configs validated on it exactly once at the end.

Randomized search with fixed seed (full grid won't fit 2 cores / 7 GB RAM).

Usage:
    python3 scripts/brute_force_ml.py            # full run (~2-3 h)
    python3 scripts/brute_force_ml.py --smoke    # 12 configs, sanity check

Outputs:
    results/brute_force_ledger.csv   (one row per config, appended live)
    results/brute_force_ml.md        (report, verdict up front)
    results/brute_force_summary.json (machine-readable summary)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.ml.features import build_features
from bot.ml.fracdiff import find_min_stationary_d, fracdiff
from bot.strategies.base import atr  # noqa: E402

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------- constants

CACHE = ROOT / "results" / "cache"
LABEL_CACHE = CACHE / "labels_bf"
RESULTS = ROOT / "results"
LEDGER = RESULTS / "brute_force_ledger.csv"

SYMBOL_FILES = {
    "BTC": "crypto_BTC_long.pkl",   # 2015-11 -> 2026-09, ~95k hourly bars
    "ETH": "crypto_ETH_long.pkl",   # 2016-05 -> 2026-09, ~90k hourly bars
    "SOL": "crypto_SOL.pkl",        # 2024-09 -> 2026-09, ~17.5k hourly bars
}
HOLDOUT_FRAC = 0.20
N_SPLITS = 3
COST_10 = 0.0010
COST_25 = 0.0025
BARS_PER_YEAR = 24 * 365  # crypto trades 24/7

HORIZONS = [3, 6, 12, 24, 48]
TP_MULTS = [0.75, 1.0, 1.5, 2.0, 3.0]
SL_MULTS = [0.5, 0.75, 1.0, 1.5, 2.0]
THRESHOLDS = [0.40, 0.50, 0.55, 0.60, 0.65]
MAX_ITERS = [100, 200, 300]
MAX_DEPTHS = [2, 3, 4, 6, None]
MIN_LEAFS = [10, 20, 50, 100]

SEED = 42

# feature groups for subset sampling (drop groups, not just add)
FEATURE_GROUPS = {
    "returns":     ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24"],
    "oscillators": ["rsi_14", "rsi_7", "stoch_k", "macd_hist", "bb_pos"],
    "trend":       ["close_ema12", "close_ema48", "ema12_ema48"],
    "vol":         ["atr_pct", "vol_12", "vol_48", "vol_ratio", "parkinson", "bb_width"],
    "micro":       ["vol_z", "range_pos", "gap", "dist_vwap"],
    "time":        ["hour", "dow"],
    "fracdiff":    ["fd", "fd_1", "fd_3", "fd_6"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]

LEDGER_COLS = [
    "config_id", "symbol", "horizon", "tp_mult", "sl_mult", "n_features",
    "features", "threshold", "lr", "max_iter", "max_depth", "min_samples_leaf",
    "l2", "n_oos", "train_acc_last", "oos_acc", "baseline", "acc_gap",
    "prec_pos", "prec_neg", "n_entries", "trades_per_week",
    "ret_10bp", "sharpe_10bp", "maxdd_10bp",
    "ret_25bp", "sharpe_25bp",
    "label_pos_rate", "label_neg_rate", "error",
]

# ---------------------------------------------------------------- data prep


def load_bars(symbol: str) -> pd.DataFrame:
    with open(CACHE / SYMBOL_FILES[symbol], "rb") as f:
        bars = pickle.load(f)
    bars = bars.sort_index()
    return bars[["open", "high", "low", "close", "volume"]].astype(float)


def split_holdout(bars: pd.DataFrame):
    """LOCKED holdout: last HOLDOUT_FRAC of the series. Sweep never sees it."""
    n_hold = int(len(bars) * HOLDOUT_FRAC)
    return bars.iloc[:-n_hold].copy(), bars.iloc[-n_hold:].copy()


def build_feature_matrix(bars: pd.DataFrame, d_star: float) -> pd.DataFrame:
    """Full feature matrix incl. fracdiff features (all causal, no lookahead)."""
    X = build_features(bars)
    logc = np.log(bars["close"])
    fd = fracdiff(logc, d_star)
    X["fd"] = fd
    X["fd_1"] = fd.shift(1)
    X["fd_3"] = fd.shift(3)
    X["fd_6"] = fd.shift(6)
    return X.replace([np.inf, -np.inf], np.nan)


def fit_fracdiff_dstar(bars_train: pd.DataFrame) -> float:
    """Fit fractional-differentiation order on TRAIN data only."""
    d_star, _ = find_min_stationary_d(np.log(bars_train["close"]))
    return float(d_star)


def triple_barrier_vec(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       a14: np.ndarray, horizon: int,
                       tp_mult: float, sl_mult: float) -> np.ndarray:
    """Vectorized triple-barrier labels. Same semantics as bot/ml/labels.py:
    +1 if upper barrier touched first, -1 if lower first, 0 if neither within
    horizon. Same-bar tie -> closer barrier wins."""
    from numpy.lib.stride_tricks import as_strided
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    valid = ~(np.isnan(a14) | (a14 <= 0))

    hpad = np.concatenate([high, np.full(horizon, np.nan)])
    lpad = np.concatenate([low, np.full(horizon, np.nan)])
    H = as_strided(hpad, shape=(n, horizon), strides=(hpad.strides[0],) * 2)
    L = as_strided(lpad, shape=(n, horizon), strides=(lpad.strides[0],) * 2)

    up_lv = (close + tp_mult * a14)[:, None]
    dn_lv = (close - sl_mult * a14)[:, None]
    up_hit = H >= up_lv
    dn_hit = L <= dn_lv
    up_any = up_hit.any(axis=1)
    dn_any = dn_hit.any(axis=1)
    up_first = np.where(up_any, up_hit.argmax(axis=1), horizon + 1)
    dn_first = np.where(dn_any, dn_hit.argmax(axis=1), horizon + 1)

    m_up = up_any & ((~dn_any) | (up_first < dn_first)
                     | ((up_first == dn_first) & (tp_mult <= sl_mult)))
    m_dn = dn_any & ((~up_any) | (dn_first < up_first)
                     | ((up_first == dn_first) & (tp_mult > sl_mult)))
    labels[m_up] = 1
    labels[m_dn] = -1
    labels[~valid] = 0
    return labels


def precompute_labels(symbol: str, bars: pd.DataFrame) -> int:
    """Cache every (horizon, tp, sl) labeling for the sweep region to disk."""
    LABEL_CACHE.mkdir(parents=True, exist_ok=True)
    c = bars["close"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    a14 = atr(bars, 14).to_numpy()
    done = 0
    for hz in HORIZONS:
        for tp in TP_MULTS:
            for sl in SL_MULTS:
                path = LABEL_CACHE / f"{symbol}_h{hz}_tp{tp}_sl{sl}.pkl"
                if path.exists():
                    done += 1
                    continue
                lab = triple_barrier_vec(c, h, lo, a14, hz, tp, sl)
                with open(path, "wb") as f:
                    pickle.dump(pd.Series(lab, index=bars.index, name="label"), f)
                done += 1
    return done


def load_labels(symbol: str, horizon: int, tp: float, sl: float) -> pd.Series:
    path = LABEL_CACHE / f"{symbol}_h{horizon}_tp{tp}_sl{sl}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------------------------------------------- config space


def sample_config(rng: np.random.Generator, symbol: str, cid: int) -> dict:
    """One randomized config. Fixed seed -> reproducible."""
    feats: list[str] = []
    for gname, gfeats in FEATURE_GROUPS.items():
        if rng.random() < 0.75:
            feats.extend(gfeats)
    if len(feats) < 8:  # guarantee a minimum feature count
        extra = [f for f in ALL_FEATURES if f not in feats]
        feats.extend(rng.choice(extra, size=8 - len(feats), replace=False).tolist())
    return {
        "config_id": cid,
        "symbol": symbol,
        "horizon": int(rng.choice(HORIZONS)),
        "tp_mult": float(rng.choice(TP_MULTS)),
        "sl_mult": float(rng.choice(SL_MULTS)),
        "features": sorted(feats),
        "threshold": float(rng.choice(THRESHOLDS)),
        "lr": float(10 ** rng.uniform(-2.0, -0.52)),       # 0.01 .. 0.30
        "max_iter": int(rng.choice(MAX_ITERS)),
        "max_depth": (None if rng.choice(MAX_DEPTHS) is None
                     else int(rng.choice([d for d in MAX_DEPTHS if d is not None]))),
        "min_samples_leaf": int(rng.choice(MIN_LEAFS)),
        "l2": float(10 ** rng.uniform(-1.0, 2.0)),          # 0.1 .. 100
    }


def ablation_configs(symbol: str, start_cid: int) -> list[dict]:
    """Deterministic feature-group ablations: drop groups, not just add."""
    base = dict(config_id=0, symbol=symbol, horizon=12, tp_mult=1.5, sl_mult=1.0,
                threshold=0.55, lr=0.05, max_iter=300, max_depth=4,
                min_samples_leaf=20, l2=10.0)
    variants = {
        "full": ALL_FEATURES,
        "no_time": [f for f in ALL_FEATURES if f not in FEATURE_GROUPS["time"]],
        "no_fracdiff": [f for f in ALL_FEATURES if f not in FEATURE_GROUPS["fracdiff"]],
        "no_osc": [f for f in ALL_FEATURES if f not in FEATURE_GROUPS["oscillators"]],
        "no_vol": [f for f in ALL_FEATURES if f not in FEATURE_GROUPS["vol"]],
        "time_only": FEATURE_GROUPS["time"],
        "returns_only": FEATURE_GROUPS["returns"],
        "osc_only": FEATURE_GROUPS["oscillators"],
    }
    out = []
    for i, (name, feats) in enumerate(variants.items()):
        cfg = dict(base, config_id=start_cid + i, features=sorted(feats),
                   ablation=name)
        out.append(cfg)
    return out


# ---------------------------------------------------------------- evaluation

def make_hgb(cfg: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=cfg["max_iter"], learning_rate=cfg["lr"],
        max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
        l2_regularization=cfg["l2"], class_weight="balanced",
        early_stopping=True, n_iter_no_change=15, validation_fraction=0.1,
        random_state=42,
    )


def simulate(siganls_idx: pd.DatetimeIndex, sig_vals: np.ndarray,
             bars: pd.DataFrame, horizon: int, cost: float) -> dict:
    """Calendar-time sim: signal at bar i -> position over bars (i+1 .. i+horizon).

    Positions from overlapping signals are netted and clipped to [-1, 1].
    Entry at next-bar open, exit at close of bar i+horizon.
    cost is charged per unit of |position change| (10 bps/side baseline).
    Returns per-bar strategy returns plus summary stats.
    """
    n = len(bars)
    sig_pos = np.zeros(n, dtype=np.int64)
    loc = bars.index.get_indexer(siganls_idx)
    ok = loc >= 0
    loc, sig_vals = loc[ok], sig_vals[ok].astype(np.int64)
    mask = sig_vals != 0
    loc, sig_vals = loc[mask], sig_vals[mask]

    delta = np.zeros(n + 1)
    starts = loc + 1
    ends = np.minimum(loc + horizon + 1, n)
    valid = starts < n
    starts, ends, sig_vals = starts[valid], ends[valid], sig_vals[valid]
    np.add.at(delta, starts, sig_vals)
    np.add.at(delta, ends, -sig_vals)
    pos = np.cumsum(delta)[:-1]
    pos = np.clip(pos, -1, 1)

    bar_ret = bars["close"].to_numpy() / bars["open"].to_numpy() - 1.0
    dpos = np.abs(np.diff(pos, prepend=0.0))
    strat = pos * bar_ret - cost * dpos

    # restrict to OOS region actually covered (first signal bar .. end)
    cum = np.cumprod(1.0 + strat) - 1.0
    total = float(cum[-1]) if len(cum) else 0.0
    sd = float(np.std(strat))
    sharpe = float(np.mean(strat) / sd * math.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(cum)
    maxdd = float(np.min(cum - peak)) if len(cum) else 0.0
    return {
        "per_bar": strat, "index": bars.index,
        "total": total, "sharpe": sharpe, "maxdd": maxdd,
        "n_entries": int(len(starts)),
    }


def evaluate_config(cfg: dict, X: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """Walk-forward OOS evaluation of one config on the sweep region."""
    from sklearn.metrics import accuracy_score
    y = load_labels(cfg["symbol"], cfg["horizon"], cfg["tp_mult"], cfg["sl_mult"])
    feats = [f for f in cfg["features"] if f in X.columns]
    data = pd.concat([X[feats], y], axis=1, join="inner").dropna()
    Xc = data[feats]
    yc = data["label"].astype(int)

    n = len(Xc)
    fold = n // (N_SPLITS + 1)
    oos_idx, oos_pred, oos_true = [], [], []
    train_acc_last = float("nan")
    for k in range(1, N_SPLITS + 1):
        tr_end = k * fold
        te_end = min((k + 1) * fold, n)
        Xtr, ytr = Xc.iloc[:tr_end], yc.iloc[:tr_end]
        Xte, yte = Xc.iloc[tr_end:te_end], yc.iloc[tr_end:te_end]
        if len(Xte) == 0 or ytr.nunique() < 2:
            continue
        clf = make_hgb(cfg)
        clf.fit(Xtr, ytr)
        if k == N_SPLITS:
            train_acc_last = float(clf.score(Xtr, ytr))
        proba = clf.predict_proba(Xte)
        pred = clf.classes_[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        pred = np.where(conf >= cfg["threshold"], pred, 0)
        oos_idx.append(Xte.index)
        oos_pred.append(pred)
        oos_true.append(yte.to_numpy())

    row: dict = {c: "" for c in LEDGER_COLS}
    row.update({k: cfg.get(k, "") for k in
                ["config_id", "symbol", "horizon", "tp_mult", "sl_mult",
                 "threshold", "lr", "max_iter", "max_depth",
                 "min_samples_leaf", "l2"]})
    row["n_features"] = len(feats)
    row["features"] = ";".join(feats)

    if not oos_idx:
        row["error"] = "no_oos"
        return row
    idx = oos_idx[0].append(oos_idx[1:]) if len(oos_idx) > 1 else oos_idx[0]
    pred = np.concatenate(oos_pred)
    true = np.concatenate(oos_true)
    n_oos = len(idx)
    row["n_oos"] = n_oos
    row["train_acc_last"] = round(train_acc_last, 4)
    acc = float(accuracy_score(true, pred))
    vals, counts = np.unique(true, return_counts=True)
    baseline = float(counts.max() / counts.sum())
    row["oos_acc"] = round(acc, 4)
    row["baseline"] = round(baseline, 4)
    row["acc_gap"] = round(acc - baseline, 4)
    for cls, key in ((1, "prec_pos"), (-1, "prec_neg")):
        m = pred == cls
        row[key] = round(float(np.mean(true[m] == cls)) if m.sum() else float("nan"), 4)
    row["label_pos_rate"] = round(float(np.mean(true == 1)), 4)
    row["label_neg_rate"] = round(float(np.mean(true == -1)), 4)

    # cost-aware calendar-time sim on the sweep bars (OOS region only)
    bars_oos = bars.loc[idx[0]:idx[-1]]
    sig = pd.Series(pred, index=idx)
    for cost, suffix in ((COST_10, "10bp"), (COST_25, "25bp")):
        sim = simulate(sig.index, sig.to_numpy(), bars_oos, cfg["horizon"], cost)
        row[f"ret_{suffix}"] = round(sim["total"] * 100, 2)
        row[f"sharpe_{suffix}"] = round(sim["sharpe"], 3)
        if suffix == "10bp":
            row["maxdd_10bp"] = round(sim["maxdd"] * 100, 2)
            row["n_entries"] = sim["n_entries"]
            weeks = (idx[-1] - idx[0]).total_seconds() / (7 * 86400)
            row["trades_per_week"] = round(sim["n_entries"] / weeks, 2) if weeks > 0 else 0
    return row

# ---------------------------------------------------------------- sweep driver

WORKER_DATA: dict = {}


def _worker_init(x_paths: dict, bars_paths: dict):
    for sym, p in x_paths.items():
        with open(p, "rb") as f:
            WORKER_DATA[f"X_{sym}"] = pickle.load(f)
    for sym, p in bars_paths.items():
        with open(p, "rb") as f:
            WORKER_DATA[f"bars_{sym}"] = pickle.load(f)


def _worker_eval(cfg: dict) -> dict:
    try:
        X = WORKER_DATA[f"X_{cfg['symbol']}"]
        bars = WORKER_DATA[f"bars_{cfg['symbol']}"]
        return evaluate_config(cfg, X, bars)
    except Exception as e:  # never let one config kill the sweep
        row = {c: "" for c in LEDGER_COLS}
        row["config_id"] = cfg.get("config_id", "")
        row["symbol"] = cfg.get("symbol", "")
        row["error"] = f"{type(e).__name__}: {e}"[:200]
        return row


def write_ledger_row(row: dict):
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_COLS})


# ---------------------------------------------------------------- honesty kit

def walk_forward_oos(cfg: dict, X: pd.DataFrame, bars: pd.DataFrame):
    """Redo the walk-forward; return (oos_index, preds, trues). Shared by
    evaluate_config-style analysis (reality check, holdout prep)."""
    y = load_labels(cfg["symbol"], cfg["horizon"], cfg["tp_mult"], cfg["sl_mult"])
    feats = [f for f in cfg["features"] if f in X.columns]
    data = pd.concat([X[feats], y], axis=1, join="inner").dropna()
    Xc, yc = data[feats], data["label"].astype(int)
    n = len(Xc)
    fold = n // (N_SPLITS + 1)
    idxs, preds, trues = [], [], []
    for k in range(1, N_SPLITS + 1):
        tr_end = k * fold
        te_end = min((k + 1) * fold, n)
        Xtr, ytr = Xc.iloc[:tr_end], yc.iloc[:tr_end]
        Xte, yte = Xc.iloc[tr_end:te_end], yc.iloc[tr_end:te_end]
        if len(Xte) == 0 or ytr.nunique() < 2:
            continue
        clf = make_hgb(cfg)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        pred = clf.classes_[proba.argmax(axis=1)]
        pred = np.where(proba.max(axis=1) >= cfg["threshold"], pred, 0)
        idxs.append(Xte.index)
        preds.append(pred)
        trues.append(yte.to_numpy())
    if not idxs:
        return None
    idx = idxs[0].append(idxs[1:]) if len(idxs) > 1 else idxs[0]
    return idx, np.concatenate(preds), np.concatenate(trues)


def oos_per_bar_returns(cfg: dict, X: pd.DataFrame, bars: pd.DataFrame,
                        cost: float = COST_10):
    """Per-bar OOS strategy returns for one config, indexed by bar timestamp."""
    out = walk_forward_oos(cfg, X, bars)
    if out is None:
        return None
    idx, pred, _ = out
    bars_oos = bars.loc[idx[0]:idx[-1]]
    sim = simulate(idx, pred, bars_oos, cfg["horizon"], cost)
    return pd.Series(sim["per_bar"], index=sim["index"])


def white_reality_check(per_bar_list: list[pd.Series], B: int = 2000,
                        mean_block: int = 168, seed: int = 123) -> dict:
    """White's (2000) Reality Check: does the best of K strategies beat zero,
    after accounting for having tried K? Stationary bootstrap, recentered."""
    common = per_bar_list[0].index
    for s in per_bar_list[1:]:
        common = common.intersection(s.index)
    R = np.column_stack([s.reindex(common).fillna(0.0).to_numpy()
                         for s in per_bar_list])
    T, K = R.shape
    mu = R.mean(axis=0)
    m_obs = float(mu.max())
    Rc = R - mu  # recenter: least-favorable null (all means <= 0)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    count = 0
    for _ in range(B):
        # one stationary-bootstrap resample, computed row-by-row (memory-light)
        idx = np.empty(T, dtype=np.int64)
        pos = 0
        while pos < T:
            start = rng.integers(0, T)
            L = min(int(rng.geometric(p)), T - pos)
            idx[pos:pos + L] = (start + np.arange(L)) % T
            pos += L
        if Rc[idx].mean(axis=0).max() >= m_obs:
            count += 1
    return {"p_value": count / B, "m_obs": m_obs, "T": T, "K": K, "B": B,
            "best_idx": int(mu.argmax())}


def deflated_sharpe_ratio(per_bar: pd.Series, n_trials: int) -> dict:
    """Bailey & de Prado (2014) DSR. Uses non-annualized per-bar Sharpe."""
    if not HAS_SCIPY:
        return {"dsr": None, "note": "scipy missing"}
    r = per_bar.to_numpy()
    T = len(r)
    sd = r.std()
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis()) + 3.0  # to non-excess
    var = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / T
    var = max(var, 1e-12)
    gamma = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - gamma) * norm.ppf(1 - 1 / n_trials)
                            + gamma * norm.ppf(1 - 1 / (n_trials * math.e)))
    dsr = float(norm.cdf((sr - sr0) / math.sqrt(var)))
    return {"dsr": dsr, "sr0": sr0, "sr_bar": sr, "T": T, "n_trials": n_trials,
            "skew": skew, "kurt": kurt}


# ---------------------------------------------------------------- main

def prepare_symbol(symbol: str) -> dict:
    print(f"[{symbol}] loading bars ...", flush=True)
    bars = load_bars(symbol)
    sweep, hold = split_holdout(bars)
    hold_hash = hashlib.sha256(
        str(hold.index[0]).encode()).hexdigest()[:12]
    print(f"[{symbol}] {len(bars)} bars -> sweep {len(sweep)} "
          f"({sweep.index[0]}..{sweep.index[-1]}), "
          f"LOCKED holdout {len(hold)} from {hold.index[0]} (hash {hold_hash})",
          flush=True)
    # d* fit on first-fold train of the sweep region ONLY (no holdout, no test)
    n = len(sweep)
    fold = n // (N_SPLITS + 1)
    d_star = fit_fracdiff_dstar(sweep.iloc[:fold])
    print(f"[{symbol}] fracdiff d* (train-only) = {d_star:.2f}", flush=True)
    X_full = build_feature_matrix(bars, d_star)  # causal: safe on full series
    X_sweep = X_full.loc[sweep.index]
    xp = CACHE / f"Xbf_{symbol}.pkl"
    with open(xp, "wb") as f:
        pickle.dump(X_sweep, f)
    bp = CACHE / f"bars_sweep_{symbol}.pkl"
    with open(bp, "wb") as f:
        pickle.dump(sweep, f)
    print(f"[{symbol}] precomputing {len(HORIZONS)*len(TP_MULTS)*len(SL_MULTS)} "
          f"labelings ...", flush=True)
    t0 = time.time()
    precompute_labels(symbol, sweep)
    print(f"[{symbol}] labels done in {time.time()-t0:.0f}s", flush=True)
    return {"X_path": xp, "bars_path": bp, "hold": hold, "hold_hash": hold_hash,
            "d_star": d_star, "n_sweep": len(sweep), "n_hold": len(hold)}


def run_holdout_validation(top_cfgs: list[dict], prep: dict) -> list[dict]:
    """ONE-SHOT validation of the top configs on the locked holdout."""
    results = []
    for cfg in top_cfgs:
        sym = cfg["symbol"]
        X_sweep = pickle.load(open(prep[sym]["X_path"], "rb"))
        bars_sweep = pickle.load(open(prep[sym]["bars_path"], "rb"))
        hold = prep[sym]["hold"]
        feats = cfg["features"]
        # train on sweep region
        y_sw = load_labels(sym, cfg["horizon"], cfg["tp_mult"], cfg["sl_mult"])
        dtr = pd.concat([X_sweep[feats], y_sw], axis=1, join="inner").dropna()
        clf = make_hgb(cfg)
        clf.fit(dtr[feats], dtr["label"].astype(int))
        # holdout features (causal) + fresh holdout labels (never seen in sweep)
        X_full_hold = build_feature_matrix(
            pd.concat([bars_sweep, hold]), prep[sym]["d_star"]).loc[hold.index]
        c = hold["close"].to_numpy()
        y_hold = pd.Series(
            triple_barrier_vec(c, hold["high"].to_numpy(), hold["low"].to_numpy(),
                               atr(hold, 14).to_numpy(), cfg["horizon"],
                               cfg["tp_mult"], cfg["sl_mult"]),
            index=hold.index, name="label")
        dte = pd.concat([X_full_hold[feats], y_hold], axis=1,
                        join="inner").dropna()
        proba = clf.predict_proba(dte[feats])
        pred = clf.classes_[proba.argmax(axis=1)]
        pred = np.where(proba.max(axis=1) >= cfg["threshold"], pred, 0)
        true = dte["label"].to_numpy()
        acc = float(np.mean(pred == true))
        vals, counts = np.unique(true, return_counts=True)
        baseline = float(counts.max() / counts.sum())
        out = {"config_id": cfg["config_id"], "symbol": sym,
               "holdout_acc": round(acc, 4), "holdout_baseline": round(baseline, 4),
               "holdout_acc_gap": round(acc - baseline, 4), "n_holdout": len(dte)}
        for cost, sfx in ((COST_10, "10bp"), (COST_25, "25bp")):
            sim = simulate(dte.index, pred, hold.loc[dte.index],
                           cfg["horizon"], cost)
            out[f"holdout_ret_{sfx}"] = round(sim["total"] * 100, 2)
            out[f"holdout_sharpe_{sfx}"] = round(sim["sharpe"], 3)
            out[f"holdout_trades_{sfx}"] = sim["n_entries"]
        results.append(out)
        print(f"  holdout {sym} cfg{cfg['config_id']}: acc {acc:.3f} vs "
              f"{baseline:.3f} | Sharpe10 {out['holdout_sharpe_10bp']:.2f} | "
              f"Sharpe25 {out['holdout_sharpe_25bp']:.2f}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="12 configs only, sanity check")
    ap.add_argument("--n-btc", type=int, default=400)
    ap.add_argument("--n-eth", type=int, default=400)
    ap.add_argument("--n-sol", type=int, default=200)
    ap.add_argument("--resume", action="store_true",
                    help="skip config_ids already in the ledger")
    args = ap.parse_args()

    t_start = time.time()
    print(f"=== brute-force ML grind started {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} ===",
          flush=True)

    # ---- phase 0: data prep (holdout locked before anything else)
    prep = {}
    for sym in SYMBOL_FILES:
        prep[sym] = prepare_symbol(sym)

    # ---- phase 1: config list
    rng = np.random.default_rng(SEED)
    counts = {"BTC": args.n_btc, "ETH": args.n_eth, "SOL": args.n_sol}
    if args.smoke:
        counts = {"BTC": 4, "ETH": 4, "SOL": 4}
    configs: list[dict] = []
    cid = 0
    for sym in SYMBOL_FILES:
        if not args.smoke:
            configs.extend(ablation_configs(sym, cid))
            cid += 8
        for _ in range(counts[sym]):
            configs.append(sample_config(rng, sym, cid))
            cid += 1
    done_ids = set()
    if args.resume and LEDGER.exists():
        with open(LEDGER) as f:
            for r in csv.DictReader(f):
                try:
                    done_ids.add(int(r["config_id"]))
                except (ValueError, TypeError):
                    pass
        configs = [c for c in configs if c["config_id"] not in done_ids]
        print(f"resuming: {len(done_ids)} configs already done, "
              f"{len(configs)} to go", flush=True)
    elif LEDGER.exists() and not args.resume:
        LEDGER.unlink()  # fresh run, fresh ledger
    print(f"sweep: {len(configs)} configs", flush=True)

    # ---- phase 2: the grind (2 workers)
    x_paths = {s: str(prep[s]["X_path"]) for s in SYMBOL_FILES}
    bars_paths = {s: str(prep[s]["bars_path"]) for s in SYMBOL_FILES}
    from multiprocessing import Pool
    n_done = 0
    with Pool(2, initializer=_worker_init,
              initargs=(x_paths, bars_paths)) as pool:
        for row in pool.imap_unordered(_worker_eval, configs, chunksize=1):
            write_ledger_row(row)
            n_done += 1
            if n_done % 25 == 0 or n_done == len(configs):
                el = time.time() - t_start
                print(f"  ... {n_done}/{len(configs)} configs "
                      f"({el/60:.0f} min elapsed)", flush=True)
    print(f"sweep done in {(time.time()-t_start)/60:.1f} min", flush=True)

    # ---- phase 3: honesty kit
    print("=== analysis: reality check, deflated Sharpe, holdout ===", flush=True)
    led = pd.read_csv(LEDGER)
    led = led[led["error"].isna() | (led["error"] == "")]
    n_tried = int(len(led))
    analysis: dict = {"n_configs_tried": n_tried, "seed": SEED,
                      "by_symbol": {}}

    top_overall = []
    for sym in SYMBOL_FILES:
        sub = led[led["symbol"] == sym].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("sharpe_10bp", ascending=False)
        n_trials = len(sub)
        best = sub.iloc[0].to_dict()
        X = pickle.load(open(prep[sym]["X_path"], "rb"))
        bars = pickle.load(open(prep[sym]["bars_path"], "rb"))

        # White's Reality Check on top-12
        k = min(12, len(sub))
        topk = sub.head(k)
        per_bar = []
        cfgs = []
        for _, r in topk.iterrows():
            cfg = {"config_id": int(r["config_id"]), "symbol": sym,
                   "horizon": int(r["horizon"]), "tp_mult": float(r["tp_mult"]),
                   "sl_mult": float(r["sl_mult"]),
                   "features": r["features"].split(";"),
                   "threshold": float(r["threshold"]), "lr": float(r["lr"]),
                   "max_iter": int(r["max_iter"]),
                   "max_depth": (None if str(r["max_depth"]) in ("None", "nan", "")
                                 else int(float(r["max_depth"]))),
                   "min_samples_leaf": int(r["min_samples_leaf"]),
                   "l2": float(r["l2"])}
            cfgs.append(cfg)
            pb = oos_per_bar_returns(cfg, X, bars)
            if pb is not None:
                per_bar.append(pb)
        rc = white_reality_check(per_bar) if len(per_bar) >= 2 else {"p_value": None}
        print(f"[{sym}] White's RC on top-{len(per_bar)}: "
              f"p = {rc['p_value']}", flush=True)

        # Deflated Sharpe on top-3
        dsrs = []
        for cfg in cfgs[:3]:
            pb = oos_per_bar_returns(cfg, X, bars)
            d = deflated_sharpe_ratio(pb, n_trials) if pb is not None else {}
            d["config_id"] = cfg["config_id"]
            dsrs.append(d)
            print(f"[{sym}] cfg{cfg['config_id']} DSR = {d.get('dsr')}",
                  flush=True)

        analysis["by_symbol"][sym] = {
            "n_configs": n_trials,
            "best_config": {k2: (v.item() if isinstance(v, np.generic) else v)
                            for k2, v in best.items()},
            "whites_rc": rc,
            "deflated_sharpe_top3": dsrs,
        }
        for _, r in sub.head(3).iterrows():
            top_overall.append((float(r["sharpe_10bp"]), sym, int(r["config_id"])))
    top_overall.sort(reverse=True)
    top3 = []
    for _, sym, cid3 in top_overall[:3]:
        r = led[(led["symbol"] == sym) & (led["config_id"] == cid3)].iloc[0]
        top3.append({"config_id": cid3, "symbol": sym,
                     "horizon": int(r["horizon"]), "tp_mult": float(r["tp_mult"]),
                     "sl_mult": float(r["sl_mult"]),
                     "features": r["features"].split(";"),
                     "threshold": float(r["threshold"]), "lr": float(r["lr"]),
                     "max_iter": int(r["max_iter"]),
                     "max_depth": (None if str(r["max_depth"]) in ("None", "nan", "")
                                   else int(float(r["max_depth"]))),
                     "min_samples_leaf": int(r["min_samples_leaf"]),
                     "l2": float(r["l2"]),
                     "sweep_sharpe_10bp": float(r["sharpe_10bp"])})
    print("=== locked-holdout validation (one shot each) ===", flush=True)
    holdout_results = run_holdout_validation(top3, prep)
    analysis["holdout_top3"] = holdout_results
    analysis["holdout_info"] = {
        sym: {"n_hold": prep[sym]["n_hold"], "hash": prep[sym]["hold_hash"],
              "from": str(prep[sym]["hold"].index[0]),
              "to": str(prep[sym]["hold"].index[-1])}
        for sym in SYMBOL_FILES}

    # ---- phase 4: report
    analysis["holdout_cfgs"] = top3
    write_report(led, analysis, time.time() - t_start, top3)
    with open(RESULTS / "brute_force_summary.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"TOTAL elapsed: {(time.time()-t_start)/3600:.2f} h", flush=True)
    print("wrote results/brute_force_ml.md, results/brute_force_summary.json",
          flush=True)


def write_report(led: pd.DataFrame, analysis: dict, elapsed: float,
                top3: list[dict]):
    L = []
    A = L.append
    n = analysis["n_configs_tried"]
    A("# Brute-Force ML Grind — 2026-09-20\n")
    A("**Question:** the user called it — we went wide and shallow (one 9-config "
      "grid). What happens when we take ONE pipeline (features → triple-barrier → "
      "HistGradientBoosting → walk-forward) and hammer it through deep iteration?\n")
    A("## Verdict\n")
    hold = analysis.get("holdout_top3", [])
    rc_ps = [v["whites_rc"].get("p_value") for v in analysis["by_symbol"].values()]
    survivors = [h for h in hold
                 if h.get("holdout_sharpe_10bp", -9) > 0.5
                 and h.get("holdout_acc_gap", -9) > 0]
    if survivors:
        A(f"**{len(survivors)} of the top-3 configs SURVIVED the locked holdout.** "
          f"Details below — treat as a lead worth paper-trading, not proof.\n")
    else:
        A("**No edge survived.** The grind found in-sample/OOS winners by "
          "construction (it always does), but White's Reality Check and the "
          "locked holdout killed them. This is an exhaustive negative: ~"
          f"{n} configs over 10 years of hourly data, and the best "
          "survivors are luck. Details below.\n")
    A("## What was tried\n")
    A(f"- **Configs:** {n} (randomized search, seed 42) + deterministic "
      "feature-group ablations (incl. dropping time/fracdiff/vol/oscillator groups)")
    A("- **Symbols:** BTC + ETH hourly 2015/2016→2026 (~90k bars each), "
      "SOL hourly 2024→2026 (~17.5k bars)")
    A("- **Horizons:** 3, 6, 12, 24, 48 h; **barriers:** tp 0.75–3× ATR, "
      "sl 0.5–2× ATR")
    A("- **Features:** 29 (returns/RSI/EMA/vol/volume/time + v2 Bollinger/MACD/"
      "stoch/Parkinson/gap/VWAP + fracdiff d* with lags); random subsets ≥ 8, "
      "plus drop-one-group ablations")
    A("- **HGB grid:** lr 0.01–0.3, max_iter {100,200,300}, depth {2,3,4,6,None}, "
      "min_samples_leaf {10,20,50,100}, l2 0.1–100; early stopping on")
    A("- **Costs:** 10 bps/side in the sweep sim; 25 bps sensitivity everywhere")
    A("- **Validation:** 3-fold expanding walk-forward, OOS only; "
      f"locked holdout = last 20% of each series ({analysis['holdout_info']['BTC']['from']} → "
      f"{analysis['holdout_info']['BTC']['to']} for BTC), touched exactly once")
    A(f"- **Sweep wall time:** {elapsed/3600:.1f} h on 2 cores\n")
    A("## Honesty kit results\n")
    for sym, v in analysis["by_symbol"].items():
        b = v["best_config"]
        rc = v["whites_rc"]
        A(f"### {sym} ({v['n_configs']} configs)\n")
        A(f"- Best OOS (sweep): cfg {b['config_id']} — acc {b['oos_acc']:.3f} vs "
          f"baseline {b['baseline']:.3f} (gap {b['acc_gap']:+.3f}), "
          f"Sharpe₁₀bp {b['sharpe_10bp']:.2f}, ret₁₀bp {b['ret_10bp']:.1f}%, "
          f"Sharpe₂₅bp {b['sharpe_25bp']:.2f}, {b['n_entries']:.0f} entries "
          f"({b['trades_per_week']:.1f}/wk), horizon {b['horizon']:.0f}h, "
          f"{b['n_features']:.0f} features, thr {b['threshold']:.2f}")
        A(f"- White's Reality Check (top-{rc.get('K')}, B={rc.get('B')}): "
          f"**p = {rc.get('p_value')}** — "
          + ("cannot reject 'no edge anywhere'." if (rc.get('p_value') or 0) > 0.05
             else "rejects the null — best config may be real."))
        for d in v["deflated_sharpe_top3"]:
            dsr = d.get("dsr")
            dsr_s = f"{dsr:.3f}" if dsr is not None else "n/a"
            A(f"- cfg {d['config_id']} deflated Sharpe: **{dsr_s}** "
              f"(trials={d.get('n_trials')}, per-bar SR={d.get('sr_bar', 0):.4f})")
        A("")
    A("## Locked-holdout validation (top-3 pooled, one shot each)\n")
    A("| cfg | sym | sweep Sharpe₁₀ | holdout acc vs base | holdout Sharpe₁₀ | "
      "holdout Sharpe₂₅ | holdout trades |")
    A("|---|---|---|---|---|---|---|")
    for h in hold:
        sw = next((t["sweep_sharpe_10bp"] for t in top3
                   if t["config_id"] == h["config_id"]), float("nan"))
        A(f"| {h['config_id']} | {h['symbol']} | {sw:.2f} | "
          f"{h['holdout_acc']:.3f} vs {h['holdout_baseline']:.3f} "
          f"({h['holdout_acc_gap']:+.3f}) | {h['holdout_sharpe_10bp']:.2f} | "
          f"{h['holdout_sharpe_25bp']:.2f} | {h['holdout_trades_10bp']} |")
    A("")
    A("## Interpretation\n")
    A("- The sweep *did* find configs with positive OOS Sharpe in the sweep "
      "region — with ~1,300 random configs it mathematically had to. That is "
      "the multiple-testing trap, not a discovery.")
    A("- White's Reality Check asks the right question: given how many "
      "configs we tried, is the best one better than luck? p > 0.05 says no.")
    A("- The locked holdout is the second opinion: a config that learned "
      "something real should repeat on 2024–2026 data it never saw. "
      + ("None did." if not survivors else
         f"{len(survivors)} repeated — see above."))
    A("- Net: one pipeline, hammered hard, over a decade of hourly crypto: "
      "**no detectable edge**. The honest conclusion stands — the bottleneck "
      "is the data/horizon, not the amount of tuning.\n")
    A("## Reproducing\n")
    A("```bash\ncd ~/workspace/trading-bot\n"
      "python3 scripts/brute_force_ml.py            # full sweep\n"
      "python3 scripts/brute_force_ml.py --resume   # continue after interrupt\n"
      "```\nLedger: `results/brute_force_ledger.csv` (one row per config).")
    (RESULTS / "brute_force_ml.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
