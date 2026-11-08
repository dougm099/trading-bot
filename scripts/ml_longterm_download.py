"""Download daily bars for the long-term ML experiment (read-only).

Universe: current S&P 500 constituents (Wikipedia, fetched with browser UA;
urllib gets 403). Bars: 2010-01-01 -> 2026-09-19, auto_adjust=True.
Plus SPY, ^VIX, sector ETFs. Cache: results/datasets/longterm/.
Sector map comes from Wikipedia's GICS Sector column (no yfinance .info crawl).
"""
from __future__ import annotations

import io
import os
import time

import pandas as pd
import requests
import yfinance as yf

OUT = os.path.expanduser("~/workspace/trading-bot/results/datasets/longterm")
START = "2010-01-01"
END = "2026-09-19"
EXTRA = ["SPY", "^VIX", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY",
         "XLU", "XLRE", "XLB", "XLC"]
UA = {"User-Agent": "Mozilla/5.0 (research; contact: local)"}

GICS_TO_ETF = {
    "Information Technology": "XLK", "Financials": "XLF", "Energy": "XLE",
    "Health Care": "XLV", "Industrials": "XLI", "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP", "Utilities": "XLU", "Real Estate": "XLRE",
    "Materials": "XLB", "Communication Services": "XLC",
}


def sp500_frame() -> pd.DataFrame:
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers=UA, timeout=30)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0][["Symbol", "GICS Sector"]].copy()
    df["ticker"] = df["Symbol"].str.replace(".", "-", regex=False).str.upper()
    df["sector"] = df["GICS Sector"]
    return df[["ticker", "sector"]].drop_duplicates("ticker").sort_values("ticker")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    uni = sp500_frame()
    print(f"universe: {len(uni)} tickers", flush=True)
    uni.to_csv(f"{OUT}/universe.csv", index=False)
    # sector map via GICS -> ETF (for sector_rel_mom feature)
    sec = uni.copy()
    sec["etf"] = sec["sector"].map(GICS_TO_ETF)
    sec[["ticker", "sector"]].to_csv(f"{OUT}/sectors.csv", index=False)
    sec[["ticker", "etf"]].to_csv(f"{OUT}/sector_etf.csv", index=False)
    print("unmapped sectors:", sec["etf"].isna().sum(), flush=True)

    tickers = uni["ticker"].tolist()
    all_syms = tickers + [e for e in EXTRA if e not in tickers]
    print(f"downloading {len(all_syms)} symbols {START}..{END} ...", flush=True)
    t0 = time.time()
    data = yf.download(all_syms, start=START, end=END, auto_adjust=True,
                       threads=True, progress=False, timeout=60)
    print(f"downloaded in {time.time()-t0:.0f}s, shape={data.shape}", flush=True)
    data = data.loc[:, (["Open", "High", "Low", "Close", "Volume"], slice(None))]
    data.to_parquet(f"{OUT}/bars.parquet")
    print("saved bars.parquet", flush=True)


if __name__ == "__main__":
    main()
