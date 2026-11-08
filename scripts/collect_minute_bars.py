"""5-minute bar collector -> results/datasets/minute_5m.csv.

STATUS 2026-09-20: BLOCKED. Stooq's download endpoint now serves a
JavaScript browser-verification challenge to scripted requests, so no
keyless minute-bar source is reachable from this machine. Supported path:
Alpaca paper minute bars once ALPACA_API_KEY/SECRET are in the Secure Vault
(broker client already exists at bot/brokers/alpaca.py). This script is kept
as the idempotent writer — point fetch() at a working source when available.

Phase-1 data for the news-impulse engine: impulse confirmation needs
minute-scale price history, which the hourly archive can't provide.
Idempotent: (symbol, ts) rows are deduplicated on every run. Best-effort —
a failed symbol never aborts the run.
"""
from __future__ import annotations

import csv
import io
import os
import sys
import time

import requests

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "datasets", "minute_5m.csv")

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "META",
           "AMZN", "GOOGL", "AMD", "JPM", "NFLX"]


def stooq_symbol(sym: str) -> str:
    return f"{sym.lower()}.us"


def fetch(sym: str) -> list:
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol(sym)}&i=5"
    try:
        r = requests.get(url, headers={"User-Agent": "research/1.0"}, timeout=30)
    except requests.RequestException as e:
        print(f"  {sym}: request failed ({e})")
        return []
    if r.status_code != 200 or not r.text.strip() or r.text.startswith("<"):
        print(f"  {sym}: bad response ({r.status_code})")
        return []
    rows = []
    try:
        reader = csv.DictReader(io.StringIO(r.text))
        for row in reader:
            try:
                rows.append({
                    "symbol": sym,
                    "ts": row["Date"] + "T" + row["Time"] + ":00Z",
                    "open": row["Open"], "high": row["High"],
                    "low": row["Low"], "close": row["Close"],
                    "volume": row["Volume"],
                })
            except (KeyError, TypeError):
                continue
    except Exception as e:  # noqa: BLE001 - never abort the run
        print(f"  {sym}: parse failed ({e})")
        return []
    return rows


def main() -> None:
    existing = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                existing.add((row["symbol"], row["ts"]))
    new_rows: list = []
    for sym in SYMBOLS:
        rows = fetch(sym)
        fresh = [r for r in rows if (r["symbol"], r["ts"]) not in existing]
        new_rows.extend(fresh)
        print(f"  {sym}: {len(rows)} fetched, {len(fresh)} new")
        time.sleep(1)  # be polite
    if new_rows:
        write_header = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["symbol", "ts", "open", "high", "low", "close", "volume"])
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    print(f"done: +{len(new_rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
