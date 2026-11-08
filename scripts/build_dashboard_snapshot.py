#!/usr/bin/env python3
"""Build the trading-bot-analytics ingest_snapshot payload from live VM files."""
import csv, json, sys
from datetime import datetime, timezone

BASE = "/home/hatch/workspace/trading-bot/results"

def num(v):
    v = (v or "").strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return None

def s(v):
    v = (v or "").strip()
    return v if v else None

ledger = []
prev_heartbeat_note = None
with open(f"{BASE}/datasets/oil_fade_paper.csv") as f:
    rows = list(csv.DictReader(f))
# Compress the 5-min poll heartbeats: keep every non-heartbeat event in full,
# and for new_day heartbeats keep only rows where the state note changed
# (armed/gap transitions) plus the first and last heartbeat of each day.
# This is lossless for state transitions; it only drops identical repeats.
last_idx = len(rows) - 1
for i, r in enumerate(rows):
    ev = r["event"].strip()
    if ev != "new_day":
        ledger.append({
            "ts_et": r["ts_et"].strip(), "date": r["date"].strip(), "event": ev,
            "side": s(r.get("side")), "qty": num(r.get("qty")), "price": num(r.get("price")),
            "status": s(r.get("status")), "note": s(r.get("note")),
        })
        continue
    note = s(r.get("note"))
    is_first = (i == 0) or (rows[i - 1]["event"].strip() != "new_day")
    is_last = (i == last_idx) or (rows[i + 1]["event"].strip() != "new_day")
    if note != prev_heartbeat_note or is_first or is_last:
        ledger.append({
            "ts_et": r["ts_et"].strip(), "date": r["date"].strip(), "event": ev,
            "side": None, "qty": None, "price": None, "status": None, "note": note,
        })
        prev_heartbeat_note = note

equity = []
with open(f"{BASE}/datasets/oil_fade_equity_d2.csv") as f:
    for r in csv.DictReader(f):
        equity.append({"date": r["date"].strip(), "equity": float(r["equity"]),
                       "drawdown_pct": float(r["drawdown_pct"])})

state = json.load(open(f"{BASE}/oil_fade_bot_state.json"))
bot_state = {
    "date": state["date"], "armed": bool(state["armed"]),
    "gap_pct": state.get("gap_pct"), "bars_seen": int(state.get("bars_seen", 0)),
    "position": state.get("position", 0), "entry": state.get("entry"), "stop": state.get("stop"),
}

study = []
with open(f"{BASE}/datasets/oil_spike_study.csv") as f:
    for r in csv.DictReader(f):
        row = {"date": r["date"].strip()}
        for k, v in r.items():
            if k == "date":
                continue
            v = (v or "").strip()
            if v.lower() in ("true", "false"):
                row[k] = v.lower() == "true"
            else:
                row[k] = num(v)
        study.append(row)

reb = json.load(open(f"{BASE}/paper_alpaca/rebalance_2026-09-18.json"))
portfolio = {
    "asof": reb["asof"], "equity": float(reb["equity"]),
    "note": reb.get("note"),
    "picks": [{"ticker": t, "score": round(float(reb["scores"].get(t, 0)), 4)} for t in reb["picks"]],
}

payload = {
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "oil_fade_ledger": ledger,
    "oil_fade_equity_backtest": equity,
    "bot_state": bot_state,
    "paper_portfolio": portfolio,
    "oil_spike_study": study,
}

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dashboard_snapshot.json"
json.dump(payload, open(out, "w"))
print(f"wrote {out}: ledger={len(ledger)} equity={len(equity)} study={len(study)} picks={len(portfolio['picks'])}")
