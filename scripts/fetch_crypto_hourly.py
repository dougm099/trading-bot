"""Fetch long-history hourly BTC/ETH bars via Coinbase public API (keyless).
Writes results/cache/crypto_BTC_long.pkl and crypto_ETH_long.pkl.
Run in background: full pull is ~94k bars/symbol (~300 candles/request).
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/trading-bot"))
from bot.data.binance import _fetch_coinbase

OUT = os.path.expanduser("~/workspace/trading-bot/results/cache")

# 2016-01-01 -> 2026-09-20 ≈ 93,900 hours; add margin
TOTAL = 95_000


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for sym in ["BTCUSDT", "ETHUSDT"]:
        print(f"Fetching {sym} hourly, target {TOTAL} bars...", flush=True)
        df = _fetch_coinbase(sym, 3600, TOTAL)
        name = sym.replace("USDT", "")
        path = os.path.join(OUT, f"crypto_{name}_long.pkl")
        df.to_pickle(path)
        print(f"{name}: {len(df)} bars {df.index[0]} -> {df.index[-1]} -> {path}", flush=True)


if __name__ == "__main__":
    main()
