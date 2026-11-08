#!/usr/bin/env python3
"""Fetch broad US equity universe + monthly bars for cross-sectional ML test (T4).

Universe source (free, keyless): Nasdaq Trader Symbol Directory
  - nasdaqtraded.txt : all NASDAQ-traded symbols
  - otherlisted.txt  : NYSE/AMEX/other-listed symbols
Filter to common stocks (ETF flag 'N', not test issues).

Survivorship note: the symbol directory reflects the 2026 listed population, so
companies that died/delisted before now are under-represented. We mitigate by
recomputing the eligible universe at each formation date from symbols with valid
price data at that date, but the selection set itself is present-day. Bias
direction: favors survivors -> disclosed in results/test_xs_ml.md.

Output: results/raw_xs/universe.parquet (monthly panel: date, ticker, adjclose,
volume, plus raw close/high/low) and results/raw_xs/universe_meta.csv.
"""
import io, os, sys, time
import requests
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "raw_xs")
os.makedirs(OUT, exist_ok=True)

def get_symbols():
    # Nasdaq screener API (free, keyless). ftp.nasdaqtrader.com is proxy-blocked
    # from this VM, so this is the fallback source. Covers US-listed common
    # stocks (stocks screener, no ETFs).
    url = ("https://api.nasdaq.com/api/screener/stocks?tableonly=true"
           "&limit=25&offset=0&download=true")
    r = requests.get(url, timeout=120, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json"})
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    df = pd.DataFrame(rows)
    df = df[df["country"].isin(["United States", ""])]
    df = df[~df["symbol"].str.contains(r"[\^/]", regex=True)]  # drop preferreds/units
    df["ticker"] = df["symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    df = df.drop_duplicates("ticker")
    meta = df[["ticker", "name", "sector", "industry", "marketCap"]].copy()
    meta["marketCap"] = pd.to_numeric(meta["marketCap"], errors="coerce")
    return meta

def main():
    meta = get_symbols()
    print(f"universe size (common stocks): {len(meta)}", flush=True)
    meta.to_csv(os.path.join(OUT, "universe_meta.csv"), index=False)

    import yfinance as yf
    tickers = meta["ticker"].tolist()
    batch, done = 400, []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i+batch]
        print(f"batch {i//batch+1}/{(len(tickers)+batch-1)//batch} ({len(chunk)} tickers)...", flush=True)
        for attempt in range(3):
            try:
                df = yf.download(chunk, period="max", interval="1mo",
                                 auto_adjust=False, group_by="ticker",
                                 threads=True, progress=False)
                break
            except Exception as e:
                print(f"  retry {attempt+1}: {e}", flush=True); time.sleep(10)
        else:
            print("  FAILED batch, skipping", flush=True); continue
        # normalize to long format
        if isinstance(df.columns, pd.MultiIndex):
            for t in chunk:
                if t not in df.columns.get_level_values(0): continue
                sub = df[t].copy()
                if sub.empty or sub["Close"].dropna().empty: continue
                sub = sub.reset_index()[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                sub["ticker"] = t
                done.append(sub)
        time.sleep(2)
    panel = pd.concat(done, ignore_index=True)
    panel = panel.rename(columns={"Date": "date", "Adj Close": "adjclose",
                                  "Close": "close", "Open": "open",
                                  "High": "high", "Low": "low", "Volume": "volume"})
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel.to_parquet(os.path.join(OUT, "universe.parquet"), index=False)
    print(f"rows: {len(panel)}, tickers with data: {panel['ticker'].nunique()}, "
          f"range: {panel['date'].min().date()}..{panel['date'].max().date()}", flush=True)

if __name__ == "__main__":
    main()
