#!/usr/bin/env python3
"""T2: fractional differentiation retrofit of the v1 ML pipeline.

Replaces the d=1 pct_change return features with fixed-width-window
fractionally-differentiated log-price features at the minimum stationary
d* (AFML Ch. 5). d* is fit on TRAIN data only, per walk-forward fold.

Everything else is the v1 protocol (results/ml_experiments.md):
  same triple-barrier labels, same 5-fold expanding walk-forward splits,
  same HistGradientBoostingClassifier, same confidence threshold,
  same simulator. Reported both gross and net of 10bps/side.

Warmup handling: the FFD window W(d*) needs up to ~1500 hourly bars of
history (d*=0.4 -> W=1458). We fetch `warmup` EXTRA bars before the
evaluation window; they are used ONLY as causal history for the transform
(never as training samples or for d* fitting). Without this, early folds
would lose their training data to the transform's burn-in.

Usage:
    python3 scripts/test_fracdiff.py --symbol BTCUSDT --bars 5000 --horizon 12
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", category=FutureWarning)

from bot.data.binance import fetch_klines
from bot.ml.features import build_features
from bot.ml.fracdiff import find_min_stationary_d, fracdiff
from bot.ml.labels import triple_barrier_labels
from bot.ml.model import train_classifier

RET_COLS = [f"ret_{n}" for n in (1, 3, 6, 12, 24)]
VOL_COLS = ["vol_12", "vol_48", "vol_ratio"]
FFD_LAGS = (1, 3, 6, 12, 24)
COST_PER_SIDE = 0.001  # 10 bps
WARMUP = 1500  # extra history bars used ONLY as causal FFD history


def build_ffd_features(logc_full: pd.Series, eval_index: pd.DatetimeIndex,
                       d_star: float) -> pd.DataFrame:
    """Causal FFD features at order d*.

    Transform runs on the FULL log-close history (warmup + eval); only the
    eval-window values are returned. f_t uses only x_s, s <= t: no lookahead.
    """
    f = fracdiff(logc_full, d_star)
    df = pd.DataFrame(index=logc_full.index)
    df["ffd_0"] = f
    for n in FFD_LAGS:
        df[f"ffd_lag{n}"] = f.shift(n)
    d1 = f.diff()
    df["ffd_vol_12"] = d1.rolling(12).std()
    df["ffd_vol_48"] = d1.rolling(48).std()
    df["ffd_vol_ratio"] = df["ffd_vol_12"] / (df["ffd_vol_48"] + 1e-12)
    return df.replace([np.inf, -np.inf], np.nan).loc[eval_index]


def run(symbol: str, bars_n: int, horizon: int, splits: int, threshold: float):
    t0 = time.time()
    print(f"Fetching {bars_n + WARMUP} hourly bars for {symbol} "
          f"({WARMUP} warmup + {bars_n} eval) ...")
    bars = fetch_klines(symbol, "1h", total=bars_n + WARMUP)
    print(f"Got {len(bars)} bars: {bars.index[0]} -> {bars.index[-1]}")
    eval_bars = bars.iloc[WARMUP:]
    print(f"Eval window: {eval_bars.index[0]} -> {eval_bars.index[-1]} "
          f"({len(eval_bars)} bars)")
    logc_full = np.log(bars["close"].astype(float))

    print("Engineering v2 features ...")
    X = build_features(eval_bars)
    print("Labeling with triple barrier ...")
    y = triple_barrier_labels(eval_bars, horizon=horizon)

    base = X.join(y.rename("label")).dropna()
    Xb, yb = base[X.columns], base["label"]
    n = len(Xb)
    fold = n // (splits + 1)
    print(f"Samples (v2, dropna): {n}; label dist: "
          f"{yb.value_counts(normalize=True).round(3).to_dict()}")

    d_stars: list[float] = []
    preds, probas, idx = [], [], []
    classes = sorted(yb.unique())
    last_fold = None

    for k in range(1, splits + 1):
        tr_end = k * fold
        te_end = min((k + 1) * fold, n)
        tr_idx = Xb.index[:tr_end]
        te_idx = Xb.index[tr_end:te_end]
        if len(te_idx) == 0 or yb.loc[tr_idx].nunique() < 2:
            continue

        # d* on TRAIN log-close only, no peeking at the test window.
        train_logc = np.log(eval_bars.loc[tr_idx, "close"].astype(float))
        d_star, diag = find_min_stationary_d(train_logc)
        d_stars.append(d_star)

        ffd = build_ffd_features(logc_full, Xb.index, d_star)
        Xall = Xb.drop(columns=RET_COLS + VOL_COLS).join(ffd).dropna()
        yall = yb.loc[Xall.index]

        Xtr = Xall.loc[tr_idx.intersection(Xall.index)]
        ytr = yall.loc[Xtr.index]
        Xte = Xall.loc[te_idx.intersection(Xall.index)]
        yte = yall.loc[Xte.index]
        if len(Xte) == 0 or ytr.nunique() < 2 or len(Xtr) == 0:
            print(f"  fold {k}: SKIPPED (train={len(Xtr)} test={len(Xte)})")
            continue
        last_fold = (Xtr, ytr, Xte, yte)
        clf = train_classifier(Xtr, ytr)
        p = clf.predict(Xte)
        pr = clf.predict_proba(Xte)
        preds.extend(p)
        probas.extend([dict(zip(clf.classes_, row)) for row in pr])
        idx.extend(Xte.index)
        n_pass = int(diag["stationary"].sum())
        print(f"  fold {k}: train={len(Xtr)} test={len(Xte)} d*={d_star:.2f} "
              f"(ADF pass: {n_pass}/{len(diag)})", flush=True)

    if not preds:
        print("No folds produced predictions — aborting.")
        return None

    pred_s = pd.Series(preds, index=idx).sort_index()
    proba_df = pd.DataFrame(probas, index=idx).sort_index()
    proba_df = proba_df.reindex(columns=[c for c in classes])

    max_proba = proba_df.max(axis=1)
    trade_pred = pred_s.where(max_proba >= threshold, 0)

    # simulator: enter next-bar open, exit `horizon` bars later (same as v1)
    b = eval_bars.loc[pred_s.index]
    strat_gross, strat_net, trades = [], [], 0
    for ts, sig in trade_pred.items():
        if sig == 0:
            continue
        try:
            i = b.index.get_loc(ts)
        except KeyError:
            continue
        if not isinstance(i, (int, np.integer)):
            continue
        if i + 1 + horizon >= len(b):
            continue
        entry = b["open"].iloc[i + 1]
        exit_ = b["close"].iloc[i + horizon]
        r = (exit_ / entry - 1) * sig
        strat_gross.append(r)
        strat_net.append(r - 2 * COST_PER_SIDE)
        trades += 1
    g = pd.Series(strat_gross)
    nn = pd.Series(strat_net)
    total_gross = float((1 + g).prod() - 1) * 100 if len(g) else 0.0
    total_net = float((1 + nn).prod() - 1) * 100 if len(nn) else 0.0
    ann = 24 * 365
    sharpe_g = float(g.mean() / g.std() * np.sqrt(ann)) if len(g) > 1 and g.std() > 0 else 0.0
    sharpe_n = float(nn.mean() / nn.std() * np.sqrt(ann)) if len(nn) > 1 and nn.std() > 0 else 0.0

    from sklearn.metrics import accuracy_score, classification_report
    y_oos = yb.loc[pred_s.index]
    acc = accuracy_score(y_oos, pred_s)
    baseline = float(y_oos.value_counts(normalize=True).max())
    report = classification_report(y_oos, pred_s, zero_division=0)

    # permutation importance on the last fold's OOS slice
    from sklearn.inspection import permutation_importance
    imp = pd.Series(dtype=float)
    if last_fold is not None:
        Xtr_i, ytr_i, Xte_i, yte_i = last_fold
        imp_clf = train_classifier(Xtr_i, ytr_i)
        perm = permutation_importance(imp_clf, Xte_i, yte_i, n_repeats=5,
                                      random_state=42, scoring="accuracy")
        imp = pd.Series(perm.importances_mean, index=Xtr_i.columns).sort_values(ascending=False)

    # memory diagnostic: does FFD preserve level memory as LdP claims?
    d_med = float(np.median(d_stars))
    f_med = fracdiff(logc_full, d_med).loc[eval_bars.index].dropna()
    lc = np.log(eval_bars["close"].astype(float)).loc[f_med.index]
    corr_ffd = float(f_med.corr(lc))
    corr_ret = float(eval_bars["close"].pct_change().loc[f_med.index].corr(lc))

    elapsed = time.time() - t0
    print(f"\n=== fracdiff walk-forward (out-of-sample), {symbol} h={horizon} ===")
    print(f"d* per fold: {[f'{d:.2f}' for d in d_stars]} (median {d_med:.2f})")
    print(f"memory check: corr(FFD@{d_med:.2f}, log-price)={corr_ffd:.3f}  "
          f"vs corr(returns, log-price)={corr_ret:.3f}")
    print(f"Accuracy:            {acc:.3f}  (majority baseline {baseline:.3f})")
    print(report)
    print(f"Sim gross:  {total_gross:+.1f}%  Sharpe {sharpe_g:+.2f}  ({trades} trades)")
    print(f"Sim net (10bps/side): {total_net:+.1f}%  Sharpe {sharpe_n:+.2f}")
    print(f"\nTop features:\n{imp.head(10).round(3).to_string()}")
    print(f"\n[{elapsed:.0f}s]")

    return {
        "symbol": symbol, "horizon": horizon, "n": n,
        "d_stars": d_stars, "d_median": d_med,
        "corr_ffd": corr_ffd, "corr_ret": corr_ret,
        "accuracy": acc, "baseline": baseline,
        "report": report,
        "gross_pct": total_gross, "sharpe_gross": sharpe_g,
        "net_pct": total_net, "sharpe_net": sharpe_n,
        "trades": trades,
        "importance": imp.head(15).round(4).to_dict(),
        "eval_start": str(eval_bars.index[0]),
        "eval_end": str(eval_bars.index[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()
    run(args.symbol, args.bars, args.horizon, args.splits, args.threshold)


if __name__ == "__main__":
    main()
