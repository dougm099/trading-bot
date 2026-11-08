#!/usr/bin/env python3
"""Forward paper-validation monitor for the VIX fear-spike signal.

LOGGING ONLY. Never places orders, never touches keys.

Protocol (pre-registered 2026-09-20):
- Run once per day after US close (e.g. via cron).
- Signal: VIX3M/VIX < 1.0 at the daily close (also logs 0.95/1.05 for context).
- A "paper episode" is opened on a signal day IF no episode is active and the
  last signal is >20 trading days old (non-overlapping episodes, same as the
  backtest in scripts/analyze_vix_extension.py).
- Intended action logged: "would buy SPY at next open, hold 20 trading days".
- Entry price is filled on the NEXT run (next day's open, from yfinance).
- Exit: close of the 20th trading day after entry. Final net return logged
  with 10 bps round-trip cost.
- Evaluation criteria (pre-registered): after n>=10 closed episodes, compute
  mean net return/trade and mean EXCESS vs SPY buy-and-hold over the identical
  hold windows. Promote only if mean excess > 0 with t > 2.0.
- Honest note: ~3 signals/year => 3-5 years to reach n=10-15. This is slow
  by construction; do not shortcut it by re-running the backtest.

State: results/vix_fear_paper_state.json
Log:   results/vix_fear_forward_log.csv
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "results" / "vix_fear_paper_state.json"
LOG = ROOT / "results" / "vix_fear_forward_log.csv"
COST_RT = 0.0010
HOLD = 20
THR = 1.0
COOLDOWN = 20  # trading days between episodes


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"episodes": [], "last_signal": None}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))


def log_row(row):
    df = pd.DataFrame([row])
    if LOG.exists():
        prev = pd.read_csv(LOG)
        if row["date"] in prev["date"].values and row["event"] in prev[prev["date"] == row["date"]]["event"].values:
            return  # already logged today
        df.to_csv(LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG, index=False)


def main():
    st = load_state()
    raw = yf.download(["^VIX", "^VIX3M", "SPY"], period="3mo", interval="1d",
                      progress=False, auto_adjust=True)
    vix = raw["Close"]["^VIX"].dropna()
    vix3m = raw["Close"]["^VIX3M"].dropna()
    spy = pd.DataFrame({"open": raw["Open"]["SPY"], "close": raw["Close"]["SPY"]}).dropna()
    idx = spy.index.intersection(vix.index).intersection(vix3m.index)
    if len(idx) == 0:
        print("no overlapping data; aborting")
        return
    last = idx[-1]
    slope = float(vix3m.loc[last] / vix.loc[last])
    today = str(last.date())
    print(f"{today}: VIX={vix.loc[last]:.2f} VIX3M={vix3m.loc[last]:.2f} slope={slope:.4f}")
    log_row({"date": today, "event": "heartbeat",
             "detail": f"daily check: no inversion (slope {slope:.3f} >= {THR})",
             "slope": round(slope, 4), "pnl": ""})

    # fill pending entries (episode opened yesterday, entry = today's open)
    for ep in st["episodes"]:
        if ep["status"] == "pending_entry":
            d = spy.index[spy.index > pd.Timestamp(ep["signal_date"])]
            if len(d):
                ep["entry_date"] = str(d[0].date())
                ep["entry_open"] = float(spy.loc[d[0], "open"])
                ep["status"] = "active"
                log_row({"date": today, "event": "paper_entry",
                         "detail": f"would buy SPY @ open {ep['entry_open']:.2f}",
                         "slope": round(slope, 4), "pnl": ""})

    # close matured episodes
    for ep in st["episodes"]:
        if ep["status"] == "active":
            entry_idx = spy.index.get_loc(pd.Timestamp(ep["entry_date"]))
            exit_pos = entry_idx + HOLD - 1
            if exit_pos < len(spy):
                exit_date = spy.index[exit_pos]
                exit_close = float(spy.loc[exit_date, "close"])
                gross = exit_close / ep["entry_open"] - 1.0
                net = gross - COST_RT
                # buy-and-hold over the same window (close before entry -> exit close)
                bh = exit_close / float(spy.iloc[entry_idx - 1]["close"]) - 1.0
                ep.update(status="closed", exit_date=str(exit_date.date()),
                          exit_close=exit_close, net_ret=round(net, 5),
                          bh_ret=round(bh, 5), excess=round(net - bh, 5))
                log_row({"date": today, "event": "paper_exit",
                         "detail": f"would sell SPY @ close {exit_close:.2f}",
                         "slope": round(slope, 4), "pnl": round(net, 5)})

    # new signal?
    active = any(e["status"] in ("pending_entry", "active") for e in st["episodes"])
    last_sig = st["last_signal"]
    cooldown_ok = True
    if last_sig:
        days_since = len(spy.index[spy.index > pd.Timestamp(last_sig)])
        cooldown_ok = days_since > COOLDOWN
    if slope < THR and not active and cooldown_ok:
        st["episodes"].append({"signal_date": today, "slope": round(slope, 4),
                               "status": "pending_entry", "asset": "SPY",
                               "hold_days": HOLD, "threshold": THR})
        st["last_signal"] = today
        log_row({"date": today, "event": "paper_signal",
                 "detail": f"VIX curve inverted ({slope:.3f}<{THR}); WOULD buy SPY at next open, hold {HOLD}d",
                 "slope": round(slope, 4), "pnl": ""})
        print("SIGNAL LOGGED (paper only): would buy SPY at next open, hold 20d")
    elif slope < THR:
        print(f"slope<{THR} but suppressed (active={active}, cooldown_ok={cooldown_ok})")

    # mark-to-market on active episodes
    for ep in st["episodes"]:
        if ep["status"] == "active":
            mtm = float(spy.loc[last, "close"]) / ep["entry_open"] - 1.0
            print(f"active paper episode: entry {ep['entry_date']} @ {ep['entry_open']:.2f}, "
                  f"unrealized {mtm:+.2%}")

    # summary of closed episodes
    closed = [e for e in st["episodes"] if e["status"] == "closed"]
    if closed:
        nets = [e["net_ret"] for e in closed]
        excs = [e["excess"] for e in closed]
        print(f"closed paper episodes: n={len(closed)} mean_net={sum(nets)/len(nets):+.3%} "
              f"mean_excess={sum(excs)/len(excs):+.3%}")

    save_state(st)
    print(f"state -> {STATE}, log -> {LOG}")


if __name__ == "__main__":
    main()
