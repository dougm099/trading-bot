#!/usr/bin/env python3
"""Train an ML trading model with walk-forward validation.

Pipeline (bioinformatics-style):
    raw bars -> features -> triple-barrier labels -> train -> walk-forward eval

Usage:
    python scripts/train_model.py --symbol BTCUSDT --bars 5000 --horizon 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.data.binance import fetch_klines
from bot.ml.features import build_features
from bot.ml.labels import triple_barrier_labels
from bot.ml.model import save_model, train_classifier, walk_forward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=12, help="label/trade horizon in bars")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()

    print(f"Fetching {args.bars} hourly bars for {args.symbol} ...")
    bars = fetch_klines(args.symbol, "1h", total=args.bars)
    print(f"Got {len(bars)} bars: {bars.index[0]} -> {bars.index[-1]}")

    print("Engineering features ...")
    X = build_features(bars)
    print("Labeling with triple barrier ...")
    y = triple_barrier_labels(bars, horizon=args.horizon)

    data = X.join(y).dropna()
    X, y = data[X.columns], data["label"]
    dist = y.value_counts(normalize=True).round(3).to_dict()
    print(f"Samples: {len(X)}, label distribution: {dist}")

    print(f"Walk-forward validation ({args.splits} splits) ...")
    res = walk_forward(X, y, bars.loc[X.index], n_splits=args.splits,
                       horizon=args.horizon, proba_threshold=args.threshold)

    print(f"\n=== walk-forward (out-of-sample) ===")
    print(f"Accuracy:            {res.accuracy:.3f}")
    print(res.report)
    print(f"Simulated strategy:  {res.strategy_return_pct:+.1f}% over {res.n_trades} trades")
    print(f"Strategy Sharpe:     {res.strategy_sharpe:.2f}")
    print(f"\nTop features:")
    print(res.feature_importance.head(10).round(3).to_string())

    print("\nTraining final model on all data ...")
    clf = train_classifier(X, y)
    out = save_model(clf, list(X.columns),
                     f"models/ml_{args.symbol}_1h.joblib",
                     meta={"symbol": args.symbol, "horizon": args.horizon,
                           "threshold": args.threshold, "wf_return": res.strategy_return_pct,
                           "wf_sharpe": res.strategy_sharpe})
    print(f"Model saved to {out}")
    print("Paper-trade it with: python scripts/run_paper.py --strategy ml --help")


if __name__ == "__main__":
    main()
