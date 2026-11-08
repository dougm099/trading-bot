"""Kraken fallback fetcher for coins where yfinance serves nothing or garbage.

Kraken public OHLC is keyless, reliable, 720 candles/request (paginated via
`since`). Covers many majors yfinance mangles (ARB, UNI, SUI, PEPE, HYPE,
TAO, JUP, ...). Writes the same yf_{SYM}.csv format (date, price, mcap,
volume_usd) into results/raw_xs_crypto/ so the test script is source-agnostic.
mcap = close x circulating_supply (CoinGecko, cached) — same proxy as yfinance.
"""
import json, os, time
import pandas as pd
import requests

RAW = os.path.expanduser("~/workspace/trading-bot/results/raw_xs_crypto")
S = requests.Session()

# symbol -> Kraken pair
PAIRS = {
    "UNI": "UNIUSD", "SUI": "SUIUSD", "PEPE": "PEPEUSD", "HYPE": "HYPEUSD",
    "TAO": "TAOUSD", "ARB": "ARBUSD", "MNT": "MNTUSD", "JUP": "JUPUSD",
    "AERO": "AEROUSD", "ASTER": "ASTERUSD", "PUMP": "PUMPUSD",
    "MORPHO": "MORPHOUSD", "TIA": "TIAUSD", "WLFI": "WLFIUSD",
    "STRK": "STRKUSD", "WLD": "WLDUSD", "WIF": "WIFUSD", "FET": "FETUSD",
    "ONDO": "ONDOUSD", "SEI": "SEIUSD", "APT": "APTUSD", "OP": "OPUSD",
    "INJ": "INJUSD", "RENDER": "RENDERUSD", "ATOM": "ATOMUSD",
}

def fetch_pair(pair, start_ts):
    out = []
    since = start_ts
    for _ in range(12):  # 12 x 720 days > 20y
        r = S.get("https://api.kraken.com/0/public/OHLC",
                  params={"pair": pair, "interval": 1440, "since": since},
                  timeout=30).json()
        if r.get("error"):
            raise RuntimeError(f"{pair}: {r['error']}")
        key = next(k for k in r["result"] if k != "last")
        rows = r["result"][key]
        if not rows:
            break
        out.extend(rows)
        since = r["result"]["last"]
        if len(rows) < 700:
            break
        time.sleep(1.2)
    return out

def main():
    info = {c["symbol"].upper(): c
            for c in json.load(open(f"{RAW}/coins_markets.json"))}
    done = 0
    for sym, pair in PAIRS.items():
        path = f"{RAW}/yf_{sym}.csv"
        if os.path.exists(path) and os.path.getsize(path) > 100:
            # refresh only if the existing file is known-garbage (handled by caller
            # deleting it first); skip otherwise
            print(f"  skip {sym} (cached)")
            continue
        try:
            rows = fetch_pair(pair, 1546300800)  # 2019-01-01
            df = pd.DataFrame(rows, columns=["t","o","h","l","close","vwap","vol","n"])
            df["date"] = pd.to_datetime(df["t"].astype(int), unit="s", utc=True)
            df = df.set_index("date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            close = df["close"].astype(float)
            vol = df["vol"].astype(float)
            px = close[close > 0]
            if len(px) < 300:
                print(f"  thin {sym}: {len(px)}")
                continue
            out = pd.DataFrame({"price": px})
            supply = (info.get(sym) or {}).get("circulating_supply") or 0
            out["mcap"] = out["price"] * supply
            out["volume_usd"] = vol.reindex(px.index).fillna(0) * out["price"]
            out.to_csv(path)
            done += 1
            print(f"  {sym}: {len(out)} days, {out.index.min().date()}..{out.index.max().date()}")
        except Exception as e:
            print(f"  FAILED {sym}: {e}")
        time.sleep(1.5)
    print(f"DONE: {done} fetched from Kraken")

if __name__ == "__main__":
    main()
