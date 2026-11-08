#!/usr/bin/env python3
"""Daily oil spike->fade study logger (prospective).

Appends one row per trading day to results/datasets/oil_spike_study.csv:
  date, tension_max, iran_mentions_24h_max, esc_6h_max, iran_headlines,
  uso_open, uso_high, uso_high_time_et, uso_close,
  spike_pct (open->high), fade_pct (high->close), day_pct (open->close)

Run once after the close on weekdays (cron: oil-spike-study, ~13:35 PT).
Idempotent: replaces any existing row for the same date.
"""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DS = BASE / "results" / "datasets"
OUT = DS / "oil_spike_study.csv"

COLS = ["date", "tension_max", "iran_mentions_24h_max", "esc_6h_max",
        "iran_headlines", "uso_open", "uso_high", "uso_high_time_et",
        "uso_close", "spike_pct", "fade_pct", "day_pct",
        "d_triggered", "d_entry", "d_exit", "d_exit_why", "d_pnl_bps"]

COST_BPS = 4.0  # paper round-trip cost


def paper_d_trade(g):
    """Paper-trade Model D: short first 5m close >= +1.5% above open,
    stop-loss +0.75% above entry, cover at close. Returns dict."""
    op = float(g["o"].iloc[0])
    sig = g[g["c"] >= op * 1.015]
    if len(sig) == 0:
        return {"d_triggered": 0, "d_entry": "", "d_exit": "",
                "d_exit_why": "", "d_pnl_bps": ""}
    entry_i = sig.index[0]
    entry = float(g.loc[entry_i, "c"])
    rest = g.loc[entry_i:].iloc[1:]
    stop = entry * 1.0075
    hit = rest[rest["h"] >= stop]
    if len(hit):
        exit_px, why = stop, "stop"
    else:
        exit_px, why = float(g["c"].iloc[-1]), "close"
    pnl = (entry - exit_px) / entry * 10000 - COST_BPS
    return {"d_triggered": 1, "d_entry": round(entry, 2),
            "d_exit": round(exit_px, 2), "d_exit_why": why,
            "d_pnl_bps": round(pnl, 1)}


def main():
    import pandas as pd
    import yfinance as yf

    d = yf.download("USO", period="2d", interval="5m", prepost=False, progress=False)
    if d is None or len(d) == 0:
        print("NO_DATA: no USO bars")
        return
    o = d["Open"].iloc[:, 0]
    h = d["High"].iloc[:, 0]
    c = d["Close"].iloc[:, 0]
    df = pd.DataFrame({"o": o, "h": h, "c": c})
    today = df.index[-1].date()
    g = df[df.index.date == today]
    if len(g) < 10:
        print(f"TOO_EARLY: only {len(g)} bars for {today}")
        return
    op = float(g["o"].iloc[0])
    hi = float(g["h"].max())
    cl = float(g["c"].iloc[-1])
    t_hi = g["h"].idxmax()

    row = {
        "date": str(today),
        "uso_open": round(op, 2),
        "uso_high": round(hi, 2),
        "uso_high_time_et": t_hi.strftime("%H:%M"),
        "uso_close": round(cl, 2),
        "spike_pct": round((hi / op - 1) * 100, 2),
        "fade_pct": round((cl / hi - 1) * 100, 2),
        "day_pct": round((cl / op - 1) * 100, 2),
    }

    row.update(paper_d_trade(g))

    # D2: same as D but skipped on gap-up days (gap > +0.5% -> momentum, don't fade)
    prior = df[df.index.date < today]
    if len(prior):
        prev_close = float(prior["c"].iloc[-1])
        gap = (op / prev_close - 1) * 100
    else:
        gap = 0.0
    row["gap_pct"] = round(gap, 2)
    if gap > 0.5:
        row.update({"d2_triggered": 0, "d2_pnl_bps": ""})
    else:
        d2 = paper_d_trade(g)
        row.update({"d2_triggered": d2["d_triggered"], "d2_pnl_bps": d2["d_pnl_bps"]})

    try:
        iw = pd.read_csv(DS / "iran_watch.csv")
        iw["d"] = pd.to_datetime(iw["ts"]).dt.date
        t = iw[iw["d"] == today]
        if len(t):
            row["tension_max"] = int(t["tension"].max())
            row["iran_mentions_24h_max"] = int(t["iran_mentions_24h"].max())
            row["esc_6h_max"] = int(t["escalation_6h"].max())
    except Exception as e:
        print(f"WARN iran_watch: {e}")
    try:
        nw = pd.read_csv(DS / "news_wide.csv", low_memory=False)
        nw["d"] = pd.to_datetime(nw["date"]).dt.date
        mask = (nw["d"] == today) & nw["title"].str.contains("iran", case=False, na=False)
        row["iran_headlines"] = int(mask.sum())
    except Exception as e:
        print(f"WARN news_wide: {e}")

    out = pd.DataFrame([{k: row.get(k, "") for k in COLS}])
    if OUT.exists():
        prev = pd.read_csv(OUT)
        prev = prev[prev["date"].astype(str) != str(today)]
        out = pd.concat([prev, out], ignore_index=True)
    out.to_csv(OUT, index=False)
    print("LOGGED", {k: row.get(k, "") for k in COLS})


if __name__ == "__main__":
    main()
