"""Fetch daily FX data from yfinance and cache to results/raw_fx/.
Read-only, no keys. Run once; test script reads the cache.
"""
from __future__ import annotations
import os
import yfinance as yf

OUT = os.path.expanduser("~/workspace/trading-bot/results/raw_fx")
PAIRS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
}

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, ticker in PAIRS.items():
        h = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        h = h[["Open", "High", "Low", "Close"]].dropna()
        path = os.path.join(OUT, f"{name}_daily.csv")
        h.to_csv(path)
        print(f"{name}: {len(h)} bars {h.index[0].date()} -> {h.index[-1].date()} -> {path}")

if __name__ == "__main__":
    main()
