"""Fetch daily price/volume history for a broad crypto universe via yfinance.

Why yfinance and not CoinGecko: CoinGecko's keyless API 401s on
market_chart/range beyond ~1 year of history, and CoinPaprika's free plan
likewise caps history at 365d. yfinance gives full daily history back to each
coin's listing (keyless).

Universe = top-200 coins by current CoinGecko market cap (cached in
coins_markets.json; stablecoins/pegged derivatives excluded there).
Historical market cap is approximated as:

    mcap_hist(coin, t) = close(t) x circulating_supply_current

Circulating supply changes slowly for most majors, so RANK reconstruction is
approximately right; it is validated in the test script against known history.
Known limitation: coins with dramatic supply changes (LUNC hyperinflation)
are EXCLUDED from universe eligibility (documented in the report); this
biases against the short leg in 2022 (misses the LUNA-collapse short), i.e.
it is conservative with respect to finding short-side edge.

Caches per-coin CSVs (date, price, mcap_proxy, volume_usd) to
results/raw_xs_crypto/. Idempotent: skips coins already cached.
"""
import json, os, time
import pandas as pd
import yfinance as yf

RAW = os.path.expanduser("~/workspace/trading-bot/results/raw_xs_crypto")
os.makedirs(RAW, exist_ok=True)

# hyperinflated / unreliable-supply coins: prices kept, universe eligibility off
SUPPLY_UNRELIABLE = {"LUNC"}

# CoinGecko ids to skip entirely. Empty for now; add ids here if a coin
# needs to be dropped from the universe for reasons beyond supply data.
EXCLUDE_IDS = set()

def main():
    coins = json.load(open(f"{RAW}/coins_markets.json"))
    todo = []
    for c in coins:
        if c["id"] in EXCLUDE_IDS:
            continue
        sym = c["symbol"].upper()
        if sym in SUPPLY_UNRELIABLE:
            continue
        path = f"{RAW}/yf_{sym}.csv"
        if os.path.exists(path) and os.path.getsize(path) > 100:
            continue
        todo.append((sym, c.get("circulating_supply") or 0))

    print(f"{len(todo)} tickers to fetch", flush=True)
    # yfinance batch download in chunks
    CHUNK = 50
    done, thin = 0, []
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        tickers = [f"{s}-USD" for s, _ in chunk]
        try:
            d = yf.download(" ".join(tickers), start="2019-01-01",
                            end="2026-09-21", interval="1d", auto_adjust=False,
                            progress=False, threads=True)
        except Exception as e:
            print(f"  chunk {i//CHUNK} failed: {e}", flush=True)
            time.sleep(15)
            continue
        closes = d["Close"]
        vols = d["Volume"] if "Volume" in d else None
        for sym, supply in chunk:
            tkr = f"{sym}-USD"
            if tkr not in closes.columns:
                thin.append(sym)
                continue
            s = closes[tkr].dropna()
            if len(s) < 300:
                thin.append((sym, len(s)))
                continue
            df = pd.DataFrame({"price": s})
            df["mcap"] = df["price"] * supply
            if vols is not None and tkr in vols.columns:
                df["volume_usd"] = vols[tkr].reindex(s.index).fillna(0) * df["price"]
            else:
                df["volume_usd"] = 0.0
            df.to_csv(f"{RAW}/yf_{sym}.csv")
            done += 1
        print(f"  chunk {i//CHUNK}: {done} cached so far", flush=True)
        time.sleep(3)

    print(f"DONE: {done} fetched; thin/missing: {thin if len(str(thin)) < 400 else str(thin)[:400]}")

if __name__ == "__main__":
    main()
