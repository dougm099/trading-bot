#!/usr/bin/env python3
"""Parameter sweep for the oil-fade bot, with a train/test split.

Grid: trigger_pct x stop_pct x gap_max_pct on 60 trading days of USO 5m bars.
Train = first 40 days, test = last 20 days. Reports:
  - frozen D2 (1.5 / 0.75 / gap<=0.5) on train and test
  - top-5 configs by train total, and what each did on test (selection-bias demo)
  - median config (what a "typical" grid point does)
  - cost sensitivity on the frozen config

The sweep is hypothesis generation. The preregistered D2 stays frozen
regardless of what wins in-sample.
"""
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.oil_fade.sim import SimCosts, run_backtest, run_equity, summarize
from bot.oil_fade.strategy import OilFadeConfig


def load_days(n_days=60):
    d5 = yf.download("USO", period="60d", interval="5m", prepost=False, progress=False)
    df = pd.DataFrame({"o": d5["Open"].iloc[:, 0], "h": d5["High"].iloc[:, 0],
                       "l": d5["Low"].iloc[:, 0], "c": d5["Close"].iloc[:, 0]})
    df["date"] = df.index.date
    dd = yf.download("USO", period="90d", interval="1d", progress=False)["Close"].iloc[:, 0]
    dates = list(dd.index)
    prev = {dates[i + 1].date(): float(dd.iloc[i]) for i in range(len(dates) - 1)}
    days = []
    for dt, g in df.groupby("date"):
        if len(g) < 60 or dt not in prev:
            continue
        g = g.copy()
        g.attrs = {"date": dt, "prior_close": prev[dt],
                   "gap_pct": round((float(g["o"].iloc[0]) / prev[dt] - 1) * 100, 2)}
        days.append(g)
    return days[-n_days:]


def main():
    days = load_days()
    print(f"days loaded: {len(days)} ({days[0].attrs['date']} -> {days[-1].attrs['date']})")
    train, test = days[:40], days[40:]
    costs = SimCosts()

    frozen = OilFadeConfig()  # D2 frozen
    for name, subset in [("TRAIN", train), ("TEST", test)]:
        s = summarize(run_backtest(subset, frozen, costs))
        print(f"\n[frozen D2 | {name}] n={s['n']} win={s['win_rate']:.0%} "
              f"mean={s['mean_bps']:+.1f} total={s['total_bps']:+.1f} t={s['t_stat']:+.2f}")

    grid = list(itertools.product([1.0, 1.25, 1.5, 1.75, 2.0],
                                  [0.5, 0.75, 1.0, 1.5],
                                  [0.25, 0.5, 1.0, None]))
    print(f"\nsweeping {len(grid)} configs...")
    results = []
    for trig, stop, gap in grid:
        cfg = OilFadeConfig(trigger_pct=trig, stop_pct=stop, gap_max_pct=gap)
        tr = summarize(run_backtest(train, cfg, costs))
        te = summarize(run_backtest(test, cfg, costs))
        results.append({"trig": trig, "stop": stop, "gap": gap,
                        "tr_n": tr["n"], "tr_total": tr["total_bps"], "tr_mean": tr["mean_bps"],
                        "te_n": te["n"], "te_total": te["total_bps"], "te_mean": te["mean_bps"]})
    r = pd.DataFrame(results).sort_values("tr_total", ascending=False)
    med = r["tr_total"].median()
    print(f"\nmedian config train total: {med:+.1f} bps over {r['tr_n'].median():.0f} trades")
    print("\ntop-5 in-sample (train) configs -> test performance:")
    print(r.head(5)[["trig", "stop", "gap", "tr_n", "tr_total", "te_n", "te_total"]].to_string(index=False))

    # cost sensitivity on frozen config (full sample)
    print("\ncost sensitivity [frozen D2, full sample]:")
    for cbps, slip in [(4, 2.0), (8, 4.0), (12, 6.0)]:
        c = SimCosts(cost_bps=cbps, entry_slip_bps=slip, stop_slip_bps=slip + 1, exit_slip_bps=slip / 2)
        s = summarize(run_backtest(days, frozen, c))
        print(f"  cost={cbps}bps slip={slip}: n={s['n']} mean={s['mean_bps']:+.1f} total={s['total_bps']:+.1f}")

    # equity curve for frozen config, 1% risk/trade
    eq = run_equity(days, frozen, costs)
    ret = (eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1) * 100
    print(f"\nequity (1% risk/trade, frozen D2): {eq['equity'].iloc[0]:,.0f} -> "
          f"{eq['equity'].iloc[-1]:,.0f} ({ret:+.1f}%), maxDD {eq['drawdown_pct'].min():.1f}%")
    eq.to_csv(Path(__file__).resolve().parent.parent / "results" / "datasets" / "oil_fade_equity_d2.csv", index=False)
    r.to_csv(Path(__file__).resolve().parent.parent / "results" / "datasets" / "oil_fade_sweep.csv", index=False)
    print("\nwrote results/datasets/oil_fade_sweep.csv and oil_fade_equity_d2.csv")


if __name__ == "__main__":
    main()
