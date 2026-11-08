"""Year-long paper-account simulation of the long-horizon ML strategy.

What it does: creates a simulated brokerage account ($100k virtual cash),
and for each month-end in a 1-year window retrains the 42d HGB model on an
expanding window (embargo enforced: the model only ever sees data available
at that time), scores the universe, and rebalances the account to the top-25
picks at the next day's close. A second account runs the equal-weight
benchmark on the same cadence. Every fill, cost (10 bps/side), and daily
mark is logged — this is an account ledger, not a vectorized backtest.

Outputs: results/paper_year_42d/{trades.csv, equity.csv, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import ml_longterm_experiment as exp
from bot.paper_account import PaperAccount

HORIZON = 42
TOP_N = 25
COST_BPS = 10.0


def hgb():
    return HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                          learning_rate=0.05, random_state=0)


def month_picks(samp: pd.DataFrame, idx: pd.DatetimeIndex, sig: pd.Timestamp,
                feats) -> list:
    """Honest per-month fit: train only on data <= embargo cutoff, score sig."""
    cutoff = exp.embargo_cutoff(idx, sig, HORIZON)
    assert idx[idx.get_loc(cutoff) + HORIZON + 1] <= sig
    train = samp[samp["date"] <= cutoff]
    test = samp[samp["date"] == sig]
    if len(test) == 0:
        return []
    mdl = hgb()
    mdl.fit(train[feats].to_numpy(), train["label"].to_numpy())
    proba = mdl.predict_proba(test[feats].to_numpy())[:, 1]
    test = test.assign(score=proba).sort_values("score", ascending=False)
    return test["ticker"].head(TOP_N).tolist()


def valid_targets(names: list, prices: pd.Series) -> dict:
    names = [t for t in names if t in prices.index and np.isfinite(prices[t]) and prices[t] > 0]
    if not names:
        return {}
    w = 1.0 / len(names)
    return {t: w for t in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--cash", type=float, default=100_000.0)
    ap.add_argument("--out", default=os.path.join(exp.RES, "paper_year_42d"))
    args = ap.parse_args()

    closes, _, _, _ = exp.load()
    idx = closes.index
    samp = pd.read_parquet(os.path.join(exp.DATA, "samples_42d.parquet"))
    feats = [f for f in exp.FEATS if f in samp.columns]
    # forward-fill single-day data gaps for account operations (fills still
    # require a real valid price via valid_targets)
    px = closes.ffill()

    sigs = exp.signal_dates(idx)
    samp_dates = set(pd.to_datetime(samp["date"].unique()))
    # only months the feature pipeline actually scored (labels need forward
    # data, so the most recent 1-2 months have no sample rows. Scoring them
    # would require a separate feature build)
    sigs = [d for d in sigs
            if idx.get_loc(d) + 1 < len(idx) and d in samp_dates][-args.months:]
    print(f"signal months: {sigs[0].date()} .. {sigs[-1].date()} ({len(sigs)})")

    strat = PaperAccount(args.cash, COST_BPS)
    bench = PaperAccount(args.cash, COST_BPS)
    first_entry = idx[idx.get_loc(sigs[0]) + 1]
    last_day = idx[-1]
    # benchmark investable set: same spirit as exp.benchmark, real stocks
    # (no SPY/sector ETFs/VIX) with >252 days of history at the start
    excl = ("SPY", "^VIX", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
            "XLY", "XLU", "XLRE", "XLB", "XLC")
    bench_universe = [t for t in closes.columns if t not in excl
                      and closes[t].loc[:first_entry].notna().sum() > 252]
    print(f"benchmark universe: {len(bench_universe)} names")

    prev = None
    for k, sig in enumerate(sigs):
        entry = idx[idx.get_loc(sig) + 1]
        if prev is not None:
            for day in idx[(idx > prev) & (idx <= entry)]:
                pxx = px.loc[day]
                strat.mark(pxx, day)
                bench.mark(pxx, day)
        prices = px.loc[entry]
        picks = month_picks(samp, idx, sig, feats)
        n_tr = strat.rebalance(valid_targets(picks, prices), prices, entry)
        n_tr_b = bench.rebalance(valid_targets(bench_universe, prices), prices, entry)
        strat.mark(prices, entry)   # post-trade (overwrites pre-trade mark)
        bench.mark(prices, entry)
        if k == 0 or k == len(sigs) - 1 or k % 3 == 0:
            print(f"  {sig.date()}: picks={picks[:5]}... traded ${n_tr:,.0f} "
                  f"(bench ${n_tr_b:,.0f})")
        prev = entry

    for day in idx[idx > prev]:
        pxx = px.loc[day]
        strat.mark(pxx, day)
        bench.mark(pxx, day)

    os.makedirs(args.out, exist_ok=True)
    strat.trades_df().to_csv(f"{args.out}/trades.csv", index=False)
    bench.trades_df().to_csv(f"{args.out}/trades_benchmark.csv", index=False)
    eq = pd.DataFrame({"strategy": strat.equity_df()["equity"],
                       "benchmark": bench.equity_df()["equity"]})
    eq.to_csv(f"{args.out}/equity.csv")

    summ = {"window": [str(sigs[0].date()), str(sigs[-1].date())],
            "horizon": HORIZON, "top_n": TOP_N, "cost_bps": COST_BPS,
            "strategy": strat.summary(), "benchmark": bench.summary(),
            "strat_warnings": len(set(strat.warnings)),
            "bench_warnings": len(set(bench.warnings))}
    json.dump(summ, open(f"{args.out}/summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
