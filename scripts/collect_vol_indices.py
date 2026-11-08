"""Own-dataset builder: daily closes of the vol-index complex.

Why: VIX term structure (spot vs 3M vs 6M), VVIX (vol of vol), and SKEW
are free on Yahoo but nobody keeps a clean local archive; this keeps
one. Feeds future regime, VRP-timing, and fear-spike research.

One row per trading day: ^VIX, ^VIX3M, ^VIX6M, ^VVIX, ^SKEW closes.
Idempotent on date. Read-only.
"""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

OUT = os.path.expanduser("~/workspace/trading-bot/results/datasets/vol_indices.csv")
TICKERS = {"vix": "^VIX", "vix3m": "^VIX3M", "vix6m": "^VIX6M",
           "vvix": "^VVIX", "skew": "^SKEW"}


def main():
    # Use the date of the last available bar, not the run date: on weekends
    # Yahoo returns Friday's closes, and stamping them with today's date
    # created duplicate-value rows. Idempotency on the bar date handles
    # weekends/holidays naturally.
    bar_date, row = None, {}
    for name, t in TICKERS.items():
        try:
            h = yf.Ticker(t).history(period="5d", auto_adjust=True)
            if len(h):
                bar_date = h.index[-1].date().isoformat()
                row[name] = round(float(h["Close"].iloc[-1]), 2)
            else:
                row[name] = None
        except Exception:
            row[name] = None
    if bar_date is None or all(v is None for v in row.values()):
        print("vol indices: no data fetched (outage?) — nothing appended")
        return
    row = {"date": bar_date, **row}
    print("vol indices:", row)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        if (old["date"] == bar_date).any():
            print("  bar date already logged")
            return
        old = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        old.to_csv(OUT, index=False)
    else:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        pd.DataFrame([row]).to_csv(OUT, index=False)
    print(f"  +1 row -> {OUT}")


if __name__ == "__main__":
    main()
