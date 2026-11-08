"""Model training, walk-forward validation, persistence.

Walk-forward (expanding window) is the honest evaluation: the model only ever
trades on data it has never seen, exactly like live trading. Same idea as
cross-validation in bioinformatics, but time-ordered: shuffling a time
series would leak the future.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class WalkForwardResult:
    accuracy: float
    report: str
    oos_predictions: pd.Series
    oos_proba: pd.DataFrame
    strategy_return_pct: float
    strategy_sharpe: float
    n_trades: int
    feature_importance: pd.Series


def train_classifier(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=10.0, class_weight="balanced", random_state=42,
    )
    clf.fit(X, y)
    return clf


def walk_forward(X: pd.DataFrame, y: pd.Series, bars: pd.DataFrame,
                 n_splits: int = 5, horizon: int = 12,
                 proba_threshold: float = 0.45) -> WalkForwardResult:
    """Expanding-window validation. Returns out-of-sample predictions and a
    naive strategy simulation: long on pred=1, short on pred=-1, held `horizon`
    bars, entered at next-bar open."""
    n = len(X)
    fold = n // (n_splits + 1)
    preds, probas, idx = [], [], []

    for k in range(1, n_splits + 1):
        tr_end = k * fold
        te_end = min((k + 1) * fold, n)
        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        Xte = X.iloc[tr_end:te_end]
        if len(Xte) == 0 or ytr.nunique() < 2:
            continue
        clf = train_classifier(Xtr, ytr)
        p = clf.predict(Xte)
        pr = clf.predict_proba(Xte)
        preds.extend(p)
        probas.extend(pr)
        idx.extend(Xte.index)

    pred_s = pd.Series(preds, index=idx).sort_index()
    proba_df = pd.DataFrame(probas, index=idx, columns=[str(c) for c in clf.classes_]).sort_index()

    # confidence filter: only trade when the winning class is confident
    max_proba = proba_df.max(axis=1)
    confident = max_proba >= proba_threshold
    trade_pred = pred_s.where(confident, 0)

    # simulate: enter next-bar open, exit `horizon` bars later
    rets = bars["close"].pct_change()
    strat_rets = []
    trades = 0
    for ts, sig in trade_pred.items():
        if sig == 0:
            continue
        try:
            i = bars.index.get_loc(ts)
        except KeyError:
            continue
        if not isinstance(i, (int, np.integer)):
            continue  # duplicate timestamp; skip
        if i + 1 + horizon >= len(bars):
            continue
        entry = bars["open"].iloc[i + 1]
        exit_ = bars["close"].iloc[i + horizon]
        r = (exit_ / entry - 1) * sig
        strat_rets.append(r)
        trades += 1
    strat_rets = pd.Series(strat_rets)
    total = float((1 + strat_rets).prod() - 1) * 100 if len(strat_rets) else 0.0
    sharpe = float(strat_rets.mean() / strat_rets.std() * np.sqrt(24 * 365)) if len(strat_rets) > 1 and strat_rets.std() > 0 else 0.0

    # feature importance via permutation on the last fold's test slice
    # (honest: measured on data the importance-model never trained on)
    from sklearn.inspection import permutation_importance
    last_tr_end = n_splits * fold
    Xtr_i, ytr_i = X.iloc[:last_tr_end], y.iloc[:last_tr_end]
    Xte_i = X.iloc[last_tr_end:last_tr_end + fold]
    yte_i = y.iloc[last_tr_end:last_tr_end + fold]
    imp_clf = train_classifier(Xtr_i, ytr_i)
    perm = permutation_importance(imp_clf, Xte_i, yte_i, n_repeats=5,
                                  random_state=42, scoring="accuracy")
    imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    acc = accuracy_score(y.loc[pred_s.index], pred_s)
    report = classification_report(y.loc[pred_s.index], pred_s, zero_division=0)
    return WalkForwardResult(acc, report, pred_s, proba_df, total, sharpe, trades, imp)


def save_model(clf, feature_names: List[str], path: str | Path,
               meta: Dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": feature_names, "meta": meta or {}}, path)
    return path


def load_model(path: str | Path) -> Dict:
    return joblib.load(path)
