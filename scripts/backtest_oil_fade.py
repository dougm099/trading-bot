#!/usr/bin/env python3
"""Backtest battery for the oil spike-fade theory.

Theory: USO intraday spikes (especially geopolitically driven morning spikes)
fade into the close. Models tested on 60 days of 5m USO bars (regular session):

  A_T : threshold fade -- short at the first 5m close >= T% above session open,
        cover at session close. T in {1.0, 1.5, 2.0}. One trade/day max.
  B   : time fade -- short at 11:00 ET every day, cover at close.
        (structural control: isolates the unconditional intraday fade)
  D   : A_1.5 + 0.75% stop-loss above entry (tail management for escalation days)

Costs: 2 bps each way (4 bps round trip). All P&L in bps.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent.parent
DS = BASE / "results" / "datasets"
COST_BPS = 4.0  # round trip


def load_bars():
    d = yf.download("USO", period="60d", interval="5m", prepost=False, progress=False)
    o = d["Open"].iloc[:, 0]
    h = d["High"].iloc[:, 0]
    l = d["Low"].iloc[:, 0]
    c = d["Close"].iloc[:, 0]
    df = pd.DataFrame({"o": o, "h": h, "l": l, "c": c})
    df["date"] = df.index.date
    return df


def session_days(df, min_bars=60):
    days = []
    for dt, g in df.groupby("date"):
        if len(g) >= min_bars:
            days.append((dt, g))
    return days


def backtest_threshold(days, thresh, stop_pct=None):
    """Short first 5m close >= thresh above open; cover at close (or stop)."""
    trades = []
    for dt, g in days:
        op = float(g["o"].iloc[0])
        trig = op * (1 + thresh)
        sig = g[g["c"] >= trig]
        if len(sig) == 0:
            continue
        entry_i = sig.index[0]
        entry = float(g.loc[entry_i, "c"])
        rest = g.loc[entry_i:].iloc[1:]
        exit_px, exit_why = float(g["c"].iloc[-1]), "close"
        if stop_pct:
            stop = entry * (1 + stop_pct)
            hit = rest[rest["h"] >= stop]
            if len(hit):
                exit_px, exit_why = stop, "stop"
        pnl = (entry - exit_px) / entry * 10000 - COST_BPS
        mae = float((rest["h"].max() - entry) / entry * 10000) if len(rest) else 0.0
        mfe = float((entry - rest["l"].min()) / entry * 10000) if len(rest) else 0.0
        trades.append({"date": str(dt), "entry": round(entry, 2),
                       "exit": round(exit_px, 2), "exit_why": exit_why,
                       "pnl_bps": round(pnl, 1), "mae_bps": round(mae, 1),
                       "mfe_bps": round(mfe, 1)})
    return trades


def backtest_time(days, time_str="11:00"):
    trades = []
    for dt, g in days:
        bars = g[g.index.strftime("%H:%M") >= time_str]
        if len(bars) == 0:
            continue
        entry_i = bars.index[0]
        entry = float(g.loc[entry_i, "c"])
        rest = g.loc[entry_i:].iloc[1:]
        exit_px = float(g["c"].iloc[-1])
        pnl = (entry - exit_px) / entry * 10000 - COST_BPS
        mae = float((rest["h"].max() - entry) / entry * 10000) if len(rest) else 0.0
        mfe = float((entry - rest["l"].min()) / entry * 10000) if len(rest) else 0.0
        trades.append({"date": str(dt), "entry": round(entry, 2),
                       "exit": round(exit_px, 2), "exit_why": "close",
                       "pnl_bps": round(pnl, 1), "mae_bps": round(mae, 1),
                       "mfe_bps": round(mfe, 1)})
    return trades


def stats(trades):
    if not trades:
        return {"n": 0}
    p = np.array([t["pnl_bps"] for t in trades])
    cum = np.cumsum(p)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    wins = p[p > 0]
    return {
        "n": len(p),
        "win_rate": round(float((p > 0).mean()), 3),
        "mean_bps": round(float(p.mean()), 1),
        "sd_bps": round(float(p.std(ddof=1)), 1) if len(p) > 1 else 0.0,
        "total_bps": round(float(p.sum()), 1),
        "sharpe_per_trade": round(float(p.mean() / p.std(ddof=1) * np.sqrt(252)), 2) if len(p) > 1 and p.std() else 0.0,
        "profit_factor": round(float(wins.sum() / -p[p < 0].sum()), 2) if (p < 0).any() else float("inf"),
        "avg_mae_bps": round(float(np.mean([t["mae_bps"] for t in trades])), 1),
        "avg_mfe_bps": round(float(np.mean([t["mfe_bps"] for t in trades])), 1),
        "max_dd_bps": round(dd, 1),
        "worst_bps": round(float(p.min()), 1),
        "best_bps": round(float(p.max()), 1),
    }


def main():
    df = load_bars()
    days = session_days(df)
    print(f"trading days in sample: {len(days)} "
          f"({days[0][0]} -> {days[-1][0]})")

    models = {
        "A_1.0": backtest_threshold(days, 0.010),
        "A_1.5": backtest_threshold(days, 0.015),
        "A_2.0": backtest_threshold(days, 0.020),
        "B_time11": backtest_time(days, "11:00"),
        "D_1.5_stop75": backtest_threshold(days, 0.015, stop_pct=0.0075),
    }
    summary = {}
    for name, tr in models.items():
        summary[name] = stats(tr)
        pd.DataFrame(tr).to_csv(DS / f"oil_fade_trades_{name}.csv", index=False)

    print("\n" + pd.DataFrame(summary).T.to_string())
    with open(BASE / "results" / "oil_fade_backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # trade frequency check: how often does each threshold trigger?
    for t in (0.010, 0.015, 0.020):
        ntrig = sum(1 for _, g in days
                    if (g["c"] >= float(g["o"].iloc[0]) * (1 + t)).any())
        print(f"threshold {t*100:.1f}% triggers on {ntrig}/{len(days)} days")


if __name__ == "__main__":
    main()
