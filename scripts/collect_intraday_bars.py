"""Own-dataset builder: hourly OHLCV bars for a liquid watchlist.

Why: free sources (yfinance) only serve ~60 days of hourly history.
This script backfills everything available on first run, then appends
new bars daily. Over months it becomes an intraday archive no free
source provides: the raw material for future microstructure,
lead-lag, and session-effect research.

Idempotent: merges on (timestamp, symbol); weekends/holidays add
nothing. Read-only: no orders, no keys.
"""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

OUT = os.path.expanduser("~/workspace/trading-bot/results/datasets/intraday_1h.csv")
WATCH = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLF", "XLK", "XLE",
         "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD",
         "JPM", "NFLX"]
COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


def fetch() -> pd.DataFrame:
    data = yf.download(WATCH, period="60d", interval="1h", auto_adjust=True,
                       group_by="ticker", progress=False, threads=True)
    rows = []
    for sym in WATCH:
        try:
            df = data[sym].dropna(subset=["Close"])
        except (KeyError, TypeError):
            continue
        if df.empty:
            continue
        idx = df.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        for ts, (_, r) in zip(idx, df.iterrows()):
            rows.append({"timestamp": ts, "symbol": sym,
                         "open": round(float(r["Open"]), 4),
                         "high": round(float(r["High"]), 4),
                         "low": round(float(r["Low"]), 4),
                         "close": round(float(r["Close"]), 4),
                         "volume": int(r["Volume"]) if pd.notna(r["Volume"]) else 0})
    return pd.DataFrame(rows, columns=COLS)


def main():
    print("intraday 1h bars:", " ".join(WATCH))
    new = fetch()
    if new.empty:
        print("  no bars fetched (weekend/holiday or Yahoo outage?)")
        return
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        have = set(zip(old["timestamp"], old["symbol"]))
        new = new[~new.apply(lambda r: (r["timestamp"], r["symbol"]) in have, axis=1)]
        old = pd.concat([old, new], ignore_index=True)
        old = old.sort_values(["symbol", "timestamp"]).drop_duplicates(
            ["timestamp", "symbol"], keep="last")
        old.to_csv(OUT, index=False)
    else:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        new.sort_values(["symbol", "timestamp"]).to_csv(OUT, index=False)
    total = pd.read_csv(OUT)
    print(f"  +{len(new)} new bars; archive now {len(total)} rows, "
          f"{total['timestamp'].min()} -> {total['timestamp'].max()}")


if __name__ == "__main__":
    main()
