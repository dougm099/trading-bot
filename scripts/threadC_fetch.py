"""Thread C (oil/politics strategies): fetch daily data for channel diagnostics.

Read-only market data download; no orders, no keys.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/workspace/trading-bot"))
import yfinance as yf
import pandas as pd

TICKERS = ["SPY", "XLE", "USO", "UUP", "QQQ", "CL=F"]
OUT = os.path.expanduser("~/workspace/trading-bot/results/datasets/threadC_daily.csv")

df = yf.download(TICKERS, start="2015-01-01", end="2026-09-21",
                 auto_adjust=True, progress=False, group_by="ticker")
close = pd.DataFrame({t: df[t]["Close"] for t in TICKERS})
close = close.dropna(how="all")
close.index = pd.to_datetime(close.index).tz_localize(None)
close.to_csv(OUT)
print(f"saved {OUT}: rows={len(close)}, cols={list(close.columns)}, "
      f"range={close.index.min().date()}..{close.index.max().date()}")
