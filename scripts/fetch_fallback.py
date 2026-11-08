"""Coinbase + Hyperliquid fallback fetcher for coins yfinance/Kraken can't cover.

Coinbase public candles: keyless, 300 candles/request, paginated by time.
Hyperliquid info/candleSnapshot: keyless, native HYPE history.
Writes yf_{SYM}.csv (date, price, mcap, volume_usd) like the other fetchers.
"""
import json, os, time
import pandas as pd
import requests

RAW = os.path.expanduser("~/workspace/trading-bot/results/raw_xs_crypto")
S = requests.Session()

def coinbase_daily(product, start_iso, end_iso):
    out = []
    start = pd.Timestamp(start_iso, tz="UTC")
    end = pd.Timestamp(end_iso, tz="UTC")
    while start < end:
        chunk_end = min(start + pd.Timedelta(days=290), end)
        rows = None
        for attempt in range(4):
            try:
                r = S.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                          params={"start": start.isoformat(), "end": chunk_end.isoformat(),
                                  "granularity": 86400}, timeout=30)
                r.raise_for_status()
                rows = r.json()
                break
            except Exception as e:
                print(f"    chunk retry {attempt+1}: {str(e)[:60]}", flush=True)
                time.sleep(5 * (attempt + 1))
        if not rows:
            break
        out.extend(rows)
        start = chunk_end
        time.sleep(0.6)
    df = pd.DataFrame(out, columns=["t", "low", "high", "open", "close", "vol"])
    df["date"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df.set_index("date").sort_index()

def hyperliquid_daily(coin, start_ms, end_ms):
    r = S.post("https://api.hyperliquid.xyz/info",
               json={"type": "candleSnapshot",
                     "req": {"coin": coin, "interval": "1d",
                             "startTime": start_ms, "endTime": end_ms}},
               timeout=30)
    r.raise_for_status()
    rows = r.json()
    recs = []
    for c in rows:
        recs.append({"date": pd.to_datetime(c["T"], unit="ms", utc=True),
                     "close": float(c["c"]), "vol": float(c["v"])})
    df = pd.DataFrame(recs)
    return df.set_index("date").sort_index() if not df.empty else df

def save(sym, df):
    info = {c["symbol"].upper(): c
            for c in json.load(open(f"{RAW}/coins_markets.json"))}
    px = df["close"].astype(float)
    px = px[px > 0]
    if len(px) < 200:
        print(f"  thin {sym}: {len(px)}")
        return
    out = pd.DataFrame({"price": px})
    supply = (info.get(sym) or {}).get("circulating_supply") or 0
    out["mcap"] = out["price"] * supply
    out["volume_usd"] = df["vol"].astype(float).reindex(px.index).fillna(0) * out["price"]
    out.to_csv(f"{RAW}/yf_{sym}.csv")
    print(f"  {sym}: {len(out)} days {out.index.min().date()}..{out.index.max().date()}")

def main():
    # UNI, APT, WLD: full history via Coinbase
    for sym, prod, start in [("UNI", "UNI-USD", "2020-09-17"),
                             ("APT", "APT-USD", "2022-10-19"),
                             ("WLD", "WLD-USD", "2023-07-24")]:
        if os.path.exists(f"{RAW}/yf_{sym}.csv"):
            print(f"  skip {sym} (cached)")
            continue
        print(f"fetching {sym} from Coinbase...", flush=True)
        df = coinbase_daily(prod, start, "2026-09-21")
        save(sym, df)
    # HYPE: Hyperliquid native (launched 2024-11-29)
    if not os.path.exists(f"{RAW}/yf_HYPE.csv"):
        print("fetching HYPE from Hyperliquid...", flush=True)
        df = hyperliquid_daily("HYPE", 1732838400000, 1789776000000)
        if df.empty:
            print("  HYPE empty from Hyperliquid")
        else:
            save("HYPE", df)

if __name__ == "__main__":
    main()
