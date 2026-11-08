"""Public market-data fetchers. No API keys needed. Good for backtests.

Primary: Binance klines. Fallback: Coinbase Exchange candles (works from
US / restricted regions where Binance returns 451).
"""
from __future__ import annotations

import time

import pandas as pd
import requests

BINANCE = "https://api.binance.com/api/v3"
COINBASE = "https://api.exchange.coinbase.com"


def _fetch_binance(symbol: str, interval: str, total: int) -> pd.DataFrame:
    out = []
    end = None
    while len(out) < total:
        params = {"symbol": symbol, "interval": interval,
                  "limit": min(1000, total - len(out))}
        if end:
            params["endTime"] = end
        r = requests.get(f"{BINANCE}/klines", params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out = batch + out
        end = batch[0][0] - 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(out, columns=cols)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["t"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("t").sort_index()[["open", "high", "low", "close", "volume"]]


def _fetch_coinbase(symbol: str, granularity: int, total: int) -> pd.DataFrame:
    # symbol like BTCUSDT -> product BTC-USD ; coinbase returns [time, low, high, open, close, volume]
    base = symbol.replace("USDT", "").replace("USD", "")
    product = f"{base}-USD"
    out = []
    end = None
    while len(out) < total:
        params = {"granularity": granularity}
        if end:
            params["end"] = end.isoformat()
            params["start"] = (end - pd.Timedelta(seconds=granularity * 300)).isoformat()
        r = requests.get(f"{COINBASE}/products/{product}/candles", params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out = batch + out
        end = pd.to_datetime(batch[-1][0], unit="s", utc=True)
        time.sleep(0.3)
    df = pd.DataFrame(out, columns=["t", "low", "high", "open", "close", "volume"])
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset="t", keep="first")
    return df.set_index("t").sort_index()[["open", "high", "low", "close", "volume"]].tail(total)


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                 limit_per_call: int = 1000, total: int = 3000) -> pd.DataFrame:
    """Fetch up to `total` klines, oldest first. Falls back to Coinbase if
    Binance is unreachable/restricted."""
    gran = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}[interval]
    try:
        return _fetch_binance(symbol, interval, total).tail(total)
    except Exception as e:
        print(f"Binance unavailable ({e}); falling back to Coinbase public data")
        return _fetch_coinbase(symbol, gran, total)
