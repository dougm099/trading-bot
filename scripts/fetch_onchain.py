"""Fetch free on-chain/crypto-fundamental data (no API keys).
Sources: DeFiLlama, Coin Metrics community API, Hyperliquid, OKX.
Caches raw JSON/CSV under results/raw_onchain/.
"""
from __future__ import annotations
import json, time, os
import pandas as pd
import requests

OUT = os.path.expanduser("~/workspace/trading-bot/results/raw_onchain")
os.makedirs(OUT, exist_ok=True)

def get(url, params=None, timeout=30, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(5 * (i + 1)); continue
            r.raise_for_status()
        except requests.HTTPError:
            raise
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed: {url}")

def fetch_stablecoin_mcap():
    """DeFiLlama total stablecoin circulating mcap, daily, since Nov 2017."""
    r = get("https://stablecoins.llama.fi/stablecoincharts/all")
    rows = r.json()
    df = pd.DataFrame([{
        "date": pd.to_datetime(int(x["date"]), unit="s", utc=True).date(),
        "stable_mcap_usd": float(x["totalCirculatingUSD"]["peggedUSD"]),
    } for x in rows])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df.to_csv(f"{OUT}/stablecoin_mcap_daily.csv", index=False)
    print("stablecoin_mcap:", df.shape, df["date"].min(), "->", df["date"].max())
    return df

def fetch_coinmetrics(metrics=("FlowInExNtv", "FlowOutExNtv", "AdrActCnt", "CapMrktCurUSD", "SplyCur")):
    """Coin Metrics community API, BTC daily. Handles pagination via next_page_token."""
    out = []
    token = None
    while True:
        params = {"assets": "btc", "metrics": ",".join(metrics),
                  "frequency": "1d", "page_size": 10000}
        if token:
            params["next_page_token"] = token
        r = get("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics", params=params)
        d = r.json()
        out.extend(d.get("data", []))
        token = d.get("next_page_token")
        print(f"  cm page: {len(d.get('data', []))} rows, more={bool(token)}")
        if not token:
            break
        time.sleep(0.5)
    recs = []
    for row in out:
        rec = {"date": pd.to_datetime(row["time"], utc=True).date()}
        for m in metrics:
            v = row.get(m)
            rec[m] = float(v) if v is not None else float("nan")
        recs.append(rec)
    df = pd.DataFrame(recs).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df.to_csv(f"{OUT}/coinmetrics_btc_daily.csv", index=False)
    print("coinmetrics:", df.shape, df["date"].min(), "->", df["date"].max())
    return df

def fetch_bitmex_funding():
    """BitMEX XBTUSD funding rates, 8h intervals, paginated (500/call), back to 2016."""
    url = "https://www.bitmex.com/api/v1/funding"
    rows = []
    start = 0
    while True:
        r = get(url, params={"symbol": "XBTUSD", "count": 500, "start": start,
                             "reverse": "false"})
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        start += len(batch)
        if len(batch) < 500:
            break
        time.sleep(0.3)
    df = pd.DataFrame([{
        "ts": pd.to_datetime(x["timestamp"], utc=True),
        "funding_rate": float(x["fundingRate"]),
    } for x in rows])
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df.to_csv(f"{OUT}/bitmex_funding.csv", index=False)
    dt = df["ts"].diff().median()
    print("bitmex_funding:", df.shape, df["ts"].min(), "->", df["ts"].max(), "median interval:", dt)
    return df

def fetch_hl_funding():
    """Hyperliquid BTC fundingHistory, paginated in time chunks (500/call)."""
    url = "https://api.hyperliquid.xyz/info"
    start = 1514764800000  # 2018-01-01
    end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows = []
    cur = start
    while cur < end:
        chunk_end = cur + 500 * 3600 * 1000  # 500 hourly records
        r = requests.post(url, json={"type": "fundingHistory", "coin": "BTC",
                                     "startTime": cur, "endTime": min(chunk_end, end)},
                          timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            cur = chunk_end
            continue
        rows.extend(batch)
        cur = batch[-1]["time"] + 1
        time.sleep(0.2)
    df = pd.DataFrame([{
        "ts": pd.to_datetime(x["time"], unit="ms", utc=True),
        "funding_rate": float(x["fundingRate"]),
        "premium": float(x["premium"]),
    } for x in rows])
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df.to_csv(f"{OUT}/hl_funding.csv", index=False)
    dt = df["ts"].diff().median()
    print("hl_funding:", df.shape, df["ts"].min(), "->", df["ts"].max(), "median interval:", dt)
    return df

def fetch_btc_daily():
    """BTC daily klines via the bot's own fetcher (Coinbase fallback)."""
    import sys
    sys.path.insert(0, os.path.expanduser("~/workspace/trading-bot"))
    from bot.data.binance import fetch_klines
    df = fetch_klines("BTCUSDT", "1d", total=3500)
    df = df.reset_index().rename(columns={"t": "ts"})
    df["date"] = pd.to_datetime(df["ts"], utc=True).dt.date
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df.to_csv(f"{OUT}/btc_daily.csv", index=False)
    print("btc_daily:", df.shape, df["date"].min(), "->", df["date"].max())
    return df

if __name__ == "__main__":
    fetch_stablecoin_mcap()
    fetch_coinmetrics()
    fetch_bitmex_funding()
    fetch_hl_funding()
    fetch_btc_daily()
    print("ALL FETCHED")
