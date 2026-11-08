"""Resilient ETH hourly fetch from Coinbase public API (keyless).
Retries with backoff; saves partial progress every 20 requests so a
network failure never loses everything; resumes from partial file.
Writes results/cache/crypto_ETH_long.pkl on completion.
"""
from __future__ import annotations
import os
import time

import pandas as pd
import requests

CACHE = os.path.expanduser("~/workspace/trading-bot/results/cache")
FINAL = os.path.join(CACHE, "crypto_ETH_long.pkl")
PARTIAL = os.path.join(CACHE, "crypto_ETH_long_partial.pkl")
URL = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
GRAN = 3600


def get_candles(end=None, start=None, tries=6):
    for a in range(tries):
        try:
            params = {"granularity": GRAN}
            if end is not None:
                params["end"] = end.isoformat()
            if start is not None:
                params["start"] = start.isoformat()
            r = requests.get(URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** a
            print(f"  request failed (try {a + 1}/{tries}): {type(e).__name__}; sleeping {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError("Coinbase candles failed after retries")


def to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset="t", keep="first")
    return df.set_index("t").sort_index()[["open", "high", "low", "close", "volume"]]


def main() -> None:
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(PARTIAL):
        have = pd.read_pickle(PARTIAL)
        end = have.index.min()
        print(f"Resuming from partial: {len(have)} bars, oldest {end}", flush=True)
    else:
        have = None
        end = None
    rows = []
    n_req = 0
    while True:
        if end is not None:
            start = end - pd.Timedelta(seconds=GRAN * 300)
            batch = get_candles(end=end, start=start)
        else:
            batch = get_candles()
        if not batch:
            print("empty batch -> reached start of history", flush=True)
            break
        oldest = pd.to_datetime(batch[-1][0], unit="s", utc=True)
        if end is not None and oldest >= end:
            # no progress: exchange re-returned the boundary window
            print(f"no progress past {oldest} -> reached start of history", flush=True)
            break
        rows = batch + rows
        end = oldest
        n_req += 1
        if n_req % 20 == 0:
            df = to_df(rows)
            if have is not None:
                df = pd.concat([df, have])
                df = df[~df.index.duplicated(keep="first")].sort_index()
            df.to_pickle(PARTIAL)
            print(f"  {n_req} requests, partial saved: {len(df)} bars back to {df.index.min()}",
                  flush=True)
        time.sleep(0.4)
    df = to_df(rows)
    if have is not None:
        df = pd.concat([df, have])
        df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_pickle(FINAL)
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)
    print(f"ETH: {len(df)} bars {df.index[0]} -> {df.index[-1]} -> {FINAL}", flush=True)


if __name__ == "__main__":
    main()
