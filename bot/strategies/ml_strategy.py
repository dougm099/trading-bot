"""ML-driven strategy: loads a trained classifier and trades its predictions.

Emits buy/sell when the predicted class (+1/-1) clears the confidence
threshold, flat otherwise. The risk manager still sizes and guards every order.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ..models import Signal
from ..ml.features import build_features
from ..ml.model import load_model
from .base import Strategy


class MLStrategy(Strategy):
    name = "ml_classifier"
    backtestable = True  # honest eval comes from walk-forward, not in-sample backtest

    def __init__(self, model_path: str | Path, proba_threshold: float = 0.45):
        bundle = load_model(model_path)
        self.clf = bundle["model"]
        self.feature_names: List[str] = bundle["features"]
        self.threshold = proba_threshold
        self.meta = bundle.get("meta", {})

    def on_bar(self, symbol: str, bars: pd.DataFrame) -> List[Signal]:
        feats = build_features(bars).dropna()
        if len(feats) < 1:
            return []
        row = feats[self.feature_names].iloc[[-1]]
        pred = int(self.clf.predict(row)[0])
        proba = float(self.clf.predict_proba(row)[0].max())
        if proba < self.threshold or pred == 0:
            return []
        action = "buy" if pred == 1 else "sell"
        return [Signal(symbol, action, strength=min(1.0, proba),
                       reason=f"ml pred={pred} p={proba:.2f}")]
