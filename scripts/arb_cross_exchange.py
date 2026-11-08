"""Thread 1: Cross-exchange spot arbitrage feasibility.
Measures BTC/USD spreads across Coinbase, Kraken, Hyperliquid with full frictions.
Historical: hourly closes (aligned); live: simultaneous quote snapshots.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

# --- Retail fee tiers (documented assumptions) ---
# Coinbase Advanced Trade: taker 0.60%, maker 0.40%
# Kraken Pro: taker 0.40%, maker 0.25%
# Hyperliquid: taker 0.035%, maker 0.02% (base tier)
FEES = {
    "coinbase": {"taker": 0.0060, "maker": 0.0040},
    "kraken": {"taker": 0.0040, "maker": 0.0025},
    "hyperliquid": {"taker": 0.00035, "maker": 0.0002},
}
# BTC withdrawal fees (approx, venue-set): Kraken 0.0001 BTC; Coinbase dynamic network fee
KRAKEN_BTC_WITHDRAW_BTC = 0.0001
# Typical on-chain BTC confirmation: ~30-60 min for 3 confs; use 1h horizon for persistence test


def fetch_coinbase_hourly(n: int = 2000) -> pd.DataFrame:
    out, end = [], None
    while len(out) < n:
        params = {"granularity": 3600}
        if end:
            params["end"] = end.isoformat()
            params["start"] = (end - pd.Timedelta(hours=300)).isoformat()
        r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                         params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out = batch + out
        end = pd.to_datetime(batch[-1][0], unit="s", utc=True)
        time.sleep(0.25)
    df = pd.DataFrame(out, columns=["t", "low", "high", "open", "close", "volume"])
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.drop_duplicates("t").sort_values("t")
    return df.tail(n).reset_index(drop=True)


def fetch_kraken_hourly(n: int = 2000) -> pd.DataFrame:
    out = []
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": "XBTUSD", "interval": 60}, timeout=20)
    r.raise_for_status()
    data = r.json()["result"]
    key = [k for k in data if k != "last"][0]
    df = pd.DataFrame(data[key], columns=["t", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["t"] = pd.to_datetime(df["t"].astype(int), unit="s", utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df[["t", "close"]].tail(n).reset_index(drop=True)


def live_quotes() -> dict:
    """Simultaneous mid-price quotes from three venues."""
    q = {}
    r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=15).json()
    q["coinbase"] = {"bid": float(r["bid"]), "ask": float(r["ask"])}
    r = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=15).json()
    xk = [k for k in r["result"] if k != "last"][0]
    q["kraken"] = {"bid": float(r["result"][xk]["b"][0]), "ask": float(r["result"][xk]["a"][0])}
    # Hyperliquid BTC perp book mid (perp, not spot — basis confound noted)
    r = requests.post("https://api.hyperliquid.xyz/info",
                      json={"type": "l2Book", "coin": "BTC"}, timeout=15).json()
    q["hyperliquid_perp"] = {"bid": float(r["levels"][0][0]["px"]),
                             "ask": float(r["levels"][1][0]["px"])}
    return q


def main():
    print("Fetching hourly closes...")
    cb = fetch_coinbase_hourly()
    kr = fetch_kraken_hourly()
    cb = cb[["t", "close"]].rename(columns={"close": "cb"})
    kr = kr.rename(columns={"close": "kr"})
    m = pd.merge_asof(cb.sort_values("t"), kr.sort_values("t"), on="t", direction="nearest",
                      tolerance=pd.Timedelta("30min")).dropna()
    # spread as % of the lower price
    m["spread_pct"] = (m[["cb", "kr"]].max(axis=1) - m[["cb", "kr"]].min(axis=1)) \
        / m[["cb", "kr"]].min(axis=1) * 100

    # Naive round-trip frictions: buy taker on cheap venue + sell taker on expensive venue
    # + BTC withdrawal fee (Kraken 0.0001 BTC on ~1 BTC; express as % of a $10k trade)
    fee_cb_kr = (FEES["coinbase"]["taker"] + FEES["kraken"]["taker"]) * 100  # ~1.00%
    btc_price = m["cb"].iloc[-1]
    withdraw_pct = KRAKEN_BTC_WITHDRAW_BTC * btc_price / 10000 * 100  # on $10k trade
    hurdle = fee_cb_kr + withdraw_pct
    print(f"hours={len(m)} btc~${btc_price:,.0f} fee_hurdle~{fee_cb_kr:.2f}% + withdraw~{withdraw_pct:.3f}% = {hurdle:.2f}%")
    print(m["spread_pct"].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).to_string())

    gross_wins = (m["spread_pct"] > hurdle).sum()
    print(f"\nHours where GROSS spread beats fee+withdrawal hurdle: {gross_wins}/{len(m)} ({gross_wins/len(m)*100:.3f}%)")

    # Persistence: of those, does the spread still exist 1h and 2h later (transfer window)?
    m["win"] = m["spread_pct"] > hurdle
    for lag in (1, 2):
        m[f"win_lag{lag}"] = m["win"].shift(-lag).fillna(False)
    persistent = m[m["win"]]
    for lag in (1, 2):
        still = persistent[f"win_lag{lag}"].mean() if len(persistent) else float("nan")
        print(f"  of gross wins, spread still > hurdle {lag}h later: {still:.1%}")

    # Price-move risk during transfer: std of hourly BTC return vs typical spread
    m["ret"] = m["cb"].pct_change()
    print(f"\nHourly BTC return std: {m['ret'].std()*100:.3f}%  |  median spread: {m['spread_pct'].median():.3f}%")
    print("-> A 1h transfer window exposes you to ~hourly vol; the median spread is a fraction of it.")

    # Split-sample: first vs second half
    half = len(m) // 2
    for name, halfm in (("first", m.iloc[:half]), ("second", m.iloc[half:])):
        w = (halfm["spread_pct"] > hurdle).sum()
        print(f"{name} half: {w}/{len(halfm)} gross-win hours ({w/len(halfm)*100:.3f}%), "
              f"99th pct spread {halfm['spread_pct'].quantile(0.99):.3f}%")

    print("\n--- live simultaneous quotes ---")
    q = live_quotes()
    prices = {k: (v["bid"] + v["ask"]) / 2 for k, v in q.items()}
    for k, v in q.items():
        print(f"{k}: bid {v['bid']:,.1f} ask {v['ask']:,.1f}")
    lo, hi = min(prices.values()), max(prices.values())
    print(f"live max-min spread: {(hi-lo)/lo*100:.3f}% (note: HL is perp, carries funding-basis confound)")

    m[["t", "cb", "kr", "spread_pct"]].to_csv("results/arb_spreads_btc.csv", index=False)
    with open("results/arb_spread_summary.json", "w") as f:
        json.dump({
            "hours": len(m),
            "hurdle_pct": round(hurdle, 3),
            "gross_win_hours": int(gross_wins),
            "gross_win_frac": round(gross_wins / len(m), 5),
            "median_spread_pct": round(float(m["spread_pct"].median()), 4),
            "p99_spread_pct": round(float(m["spread_pct"].quantile(0.99)), 4),
            "hourly_ret_std_pct": round(float(m["ret"].std() * 100), 4),
        }, f, indent=2)
    print("\nwrote results/arb_spreads_btc.csv, results/arb_spread_summary.json")


if __name__ == "__main__":
    main()
