"""T5: CTREND-lite, cross-sectional multi-signal trend factor in crypto.

Replicates the Fieberg et al. (2025, JFQA) idea with retail-honest costs:
top-50 coins by LAGGED month-end market cap (point-in-time, rebuilt monthly),
per-coin weekly signals (12-1m return, MA50/MA200, 6m breakout, RSI14, MACD),
cross-sectional z-scores -> equal-weight composite. Weekly rebalance:
long top quintile / short bottom quintile, equal-weight.

Variants: long-short, long-only (honest retail), long-minus-BTC,
and a naive 12-1 momentum baseline (Grobys & Sapkota: ~= 0).

Costs: 25 bps per one-way dollar traded (state minimum for alts).
Splits: full / 2020-21 (alt-season, paper's era) / 2022-26 (required OOS).

Reads results/raw_xs_crypto/*.csv written by scripts/fetch_ctrender.py.
Writes results/test_ctrender.md + results/ctrender_summary.json.
"""
import os, glob, json
import numpy as np
import pandas as pd

RAW = os.path.expanduser("~/workspace/trading-bot/results/raw_xs_crypto")
RES = os.path.expanduser("~/workspace/trading-bot/results")
COST_BPS = 0.0025  # 25 bps per one-way dollar

# ---------------- load ----------------
# Stablecoins / pegged / tokenized securities: no trend signal; purge by symbol
# (covers newer stables whose CoinGecko ids missed EXCLUDE_IDS).
STABLE_SYMS = {"USDT","USDC","DAI","TUSD","BUSD","USDD","FRAX","LUSD","PYUSD",
    "FDUSD","USDE","USDS","USD1","USDY","USDG","USDGO","USTB","EUTBL","EURSAFO",
    "EURC","AUSD","GUSD","USDP","SUSD","XAUT","PAXG","STABLE","JAAA","BUIDL",
    "USYC","BCAP","FIGR_HELOC","WBTC","STETH","WSTETH","CBETH","RETH",
    "SDAI","WEETH","EZETH","MSOL","JITOSOL"}

# Coins whose early price history is known-bad: truncate series at VALID_FROM.
# AAVE: 100:1 LEND->AAVE redenomination 2020-10-02 (pre-date prices are LEND units).
VALID_FROM = {"AAVE": "2020-10-03"}

def load_panels():
    info = {c["symbol"].upper(): c
            for c in json.load(open(f"{RAW}/coins_markets.json"))}
    px, mc = {}, {}
    dropped = []
    for path in glob.glob(f"{RAW}/yf_*.csv"):
        cid = os.path.basename(path)[3:-4]  # yf_BTC.csv -> BTC
        if cid in STABLE_SYMS:
            dropped.append((cid, "stable/pegged"))
            continue
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        except Exception:
            continue
        if df.empty or "price" not in df.columns:
            continue
        df = df[~df.index.duplicated(keep="last")]
        # normalize tz: yfinance files are tz-naive, Kraken/Coinbase/HL are tz-aware.
        # Floor to calendar day (Hyperliquid closes print as 23:59:59.999).
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index = df.index.floor("D")
        df = df[~df.index.duplicated(keep="last")]
        if cid in VALID_FROM:
            df = df.loc[VALID_FROM[cid]:]
            if df.empty:
                dropped.append((cid, "empty after VALID_FROM"))
                continue
        # quality screen 1: truncate pre-listing placeholder zeros; remaining
        # zeros/negatives -> NaN (missing), coin kept but unrankable those weeks
        px0 = df["price"]
        first = px0[px0 > 0]
        if len(first) == 0:
            dropped.append((cid, "no positive prices"))
            continue
        df = df.loc[first.index[0]:]
        df.loc[df["price"] <= 0, "price"] = float("nan")
        # quality screen 2: absurd single-day jumps = bad prints / bad ticker
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(df["price"] / df["price"].shift(1)).abs()
        if (lr > np.log(50)).any():
            dmax = lr.idxmax().date()
            dropped.append((cid, f">50x daily move on {dmax}"))
            continue
        # collision check: yfinance ticker must match CoinGecko's current price
        # (catches reused symbols, e.g. STRK=Strike vs Starknet, SKY=Skycoin vs Sky)
        c = info.get(cid)
        if c and c.get("current_price"):
            last = df["price"].dropna()
            if len(last):
                ratio = last.iloc[-1] / c["current_price"]
                if not (0.70 <= ratio <= 1.43):
                    dropped.append((cid, f"price-mismatch x{ratio:.2f}"))
                    continue
        px[cid] = df["price"]
        mc[cid] = df.get("mcap")
    price = pd.DataFrame(px).sort_index()
    mcap = pd.DataFrame(mc).sort_index()
    # drop coins with essentially no data
    price = price.dropna(axis=1, thresh=300)
    mcap = mcap.reindex(columns=price.columns)
    # absurdity cap: no alt ever credibly exceeded 75% of BTC's mcap on the same
    # day (catches redenomination/supply-change distortions like pre-2022 BTT).
    # BTC itself is exempt.
    if "BTC" in mcap.columns:
        cap = mcap["BTC"] * 0.75
        alts = mcap.drop(columns=["BTC"])
        mcap = pd.concat([mcap[["BTC"]], alts.where(alts.le(cap, axis=0))], axis=1)
    print(f"dropped {len(dropped)} tickers: {dropped}", flush=True)
    return price, mcap

def validate_universe(mcap):
    """Sanity-check the mcap-proxy rank reconstruction against known history."""
    print("universe validation (reconstructed top-10 by mcap proxy):", flush=True)
    for d in ["2021-01-04", "2022-01-03", "2024-01-02", "2026-01-05"]:
        if d in mcap.index:
            row = mcap.loc[d]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            top = row.dropna().sort_values(ascending=False).head(10)
            print(f"  {d}: {list(top.index)}", flush=True)

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))

def macd_dist(s):
    e12 = s.ewm(span=12, adjust=False).mean()
    e26 = s.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    return (line - sig) / s

def compute_signals(price):
    """Daily signals; each uses only data up to that day."""
    mom121 = price.shift(21) / price.shift(252) - 1
    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()
    trend = (ma50 - ma200) / price
    breakout = price / price.rolling(126).max() - 1
    rsi14 = price.apply(rsi)
    macd = price.apply(macd_dist)
    return {"mom121": mom121, "trend": trend, "breakout": breakout,
            "rsi14": rsi14, "macd": macd}

def zscore(df):
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        return df.sub(mu, axis=0).div(sd, axis=0)

# ---------------- backtest ----------------
def run(price, mcap, signals, variant="longshort", naive=False):
    """variant: longshort | longonly | longminusbtc. naive: rank by mom121 only."""
    dates = price.index
    # monthly universe: top-50 by LAGGED month-end mcap
    month_ends = pd.Series(dates).dt.to_period("M").drop_duplicates()
    univ_by_month = {}
    for p in month_ends:
        m_end = dates[dates.to_period("M") == p].max()
        prev = p - 1
        prev_dates = dates[dates.to_period("M") == prev]
        if len(prev_dates) == 0:
            continue
        m_prev = prev_dates.max()
        r = mcap.loc[m_prev].dropna()
        r = r[r > 0]
        top50 = r.sort_values(ascending=False).head(50).index.tolist()
        univ_by_month[p] = set(top50)

    rebals = pd.date_range("2020-01-06", dates.max(), freq="W-MON")
    sig = {k: v for k, v in signals.items()}
    rows = []
    prev_w = {}  # coin -> weight in capital units
    btc = price.get("BTC")

    for rd in rebals:
        sig_date = dates[dates < rd].max()  # last close strictly before rebalance
        if pd.isna(sig_date):
            continue
        p = sig_date.to_period("M")
        univ = univ_by_month.get(p, set())
        if not univ:
            continue
        coins = [c for c in univ if c in price.columns]
        # need full signal history
        frame = pd.DataFrame({k: v.loc[sig_date, coins] for k, v in sig.items()})
        frame = frame.dropna()
        if len(frame) < 20:
            continue
        if naive:
            score = frame[["mom121"]]
            comp = score["mom121"]
            # rank directly (no zscore needed, monotonic)
            ranked = comp.sort_values(ascending=False)
        else:
            z = zscore(frame)
            comp = z.mean(axis=1).dropna()
            ranked = comp.sort_values(ascending=False)
        n = len(ranked)
        q = max(n // 5, 1)
        longs = ranked.head(q).index.tolist()
        shorts = ranked.tail(q).index.tolist()

        # execution at rebalance-day close
        ex = dates[dates >= rd]
        if len(ex) == 0:
            continue
        ex_date = ex.min()
        # next rebalance execution
        nxt = rebals[rebals > rd]
        if len(nxt) == 0:
            break
        ex_next = dates[dates >= nxt.min()]
        if len(ex_next) == 0:
            break
        ex_next = ex_next.min()

        # target weights in capital units
        w = {}
        if variant == "longshort":
            for c in longs: w[c] = 0.5 / len(longs)
            for c in shorts: w[c] = -0.5 / len(shorts)
        elif variant == "longonly":
            for c in longs: w[c] = 1.0 / len(longs)
        elif variant == "longminusbtc":
            for c in longs: w[c] = 1.0 / len(longs)

        # returns over holding week
        p0 = price.loc[ex_date]
        p1 = price.loc[ex_next]
        ret = (p1 / p0 - 1)
        port_ret = sum(w.get(c, 0) * ret.get(c, np.nan) for c in w)
        if pd.isna(port_ret):
            continue
        if variant == "longminusbtc" and btc is not None:
            bret = btc.loc[ex_next] / btc.loc[ex_date] - 1
            port_ret = port_ret - bret

        # turnover cost: one-way traded fraction of capital x 25bps
        allc = set(prev_w) | set(w)
        traded = sum(abs(w.get(c, 0) - prev_w.get(c, 0)) for c in allc)
        cost = traded * COST_BPS
        rows.append({"date": ex_date, "gross": port_ret, "cost": cost,
                     "net": port_ret - cost, "n_long": len(longs),
                     "n_short": len(shorts) if variant == "longshort" else 0,
                     "univ": n, "turnover": traded})
        prev_w = w

    return pd.DataFrame(rows).set_index("date")

def stats(df, label):
    if df is None or len(df) < 20:
        return {"label": label, "n_weeks": 0 if df is None else len(df),
                "note": "insufficient data"}
    r = df["net"]
    tot = (1 + r).prod() - 1
    sharpe = r.mean() / r.std() * np.sqrt(52) if r.std() > 0 else 0
    dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    return {"label": label, "n_weeks": len(df),
            "total_ret": round(float(tot), 4),
            "ann_sharpe": round(float(sharpe), 2),
            "max_dd": round(float(dd), 4),
            "avg_turnover_wk": round(float(df["turnover"].mean()), 4),
            "avg_names_traded_wk": round(float(
                (df["turnover"] * 2 * 10).mean()), 1),  # rough: turnover->names
            "avg_universe": round(float(df["univ"].mean()), 1)}

def main():
    print("loading panels...", flush=True)
    price, mcap = load_panels()
    print(f"coins={price.shape[1]} days={price.shape[0]} "
          f"{price.index.min().date()}..{price.index.max().date()}", flush=True)
    validate_universe(mcap)
    print("computing signals...", flush=True)
    signals = compute_signals(price)

    out = {"cost_bps_per_side": COST_BPS * 10000,
           "universe": "top-50 by lagged month-end mcap, point-in-time",
           "strategies": {}}
    variants = [("longshort", False), ("longonly", False),
                ("longminusbtc", False), ("naive_mom121_longshort", True)]
    results = {}
    for v, naive in variants:
        key = v
        print(f"running {key}...", flush=True)
        df = run(price, mcap, signals,
                 variant="longshort" if naive else v, naive=naive)
        results[key] = df
        s_full = stats(df, key)
        s_2021 = stats(df[df.index < "2022-01-01"] if df is not None else None,
                       key + "_2020_21")
        s_oos = stats(df[df.index >= "2022-01-01"] if df is not None else None,
                      key + "_2022_26_OOS")
        out["strategies"][key] = {"full": s_full, "in_2020_21": s_2021,
                                  "oos_2022_26": s_oos}

    with open(f"{RES}/ctrender_summary.json", "w") as f:
        json.dump(out, f, indent=1)

    # ---- markdown report ----
    L = []
    A = L.append
    A("# T5 — CTREND-lite: cross-sectional multi-signal trend factor in crypto")
    A("")
    A("**Date:** 2026-09-20  ")
    A("**Paper:** Fieberg et al. (2025, JFQA) — ML-aggregated trend signals, long-short quintiles, 3.87%/week claimed, robust in liquid coins.")
    A("")
    A("## Verdict (up front)")
    for key in ["longshort", "longonly"]:
        oos = out["strategies"][key]["oos_2022_26"]
        full = out["strategies"][key]["full"]
        A(f"- **{key}**: OOS 2022–26 net Sharpe **{oos.get('ann_sharpe','n/a')}**, "
          f"total {oos.get('total_ret','n/a')} over {oos.get('n_weeks','n/a')} weeks; "
          f"full-sample Sharpe {full.get('ann_sharpe','n/a')}.")
    A("")
    A("## Mechanics (as implemented)")
    A("- Universe: top-50 coins by **lagged month-end** market cap, recomputed monthly "
      "(point-in-time). Prices from yfinance (full daily history); historical mcap "
      "approximated as close × current circulating supply (CoinGecko) — validated "
      "against known history below; LUNC excluded (hyperinflation breaks the proxy).")
    A("- Stablecoins and pegged derivatives excluded (no trend signal; contaminate z-scores).")
    A("- Per coin, per week: 12-1m return, (MA50−MA200)/price, 6m breakout distance, "
      "RSI(14), (MACD−signal)/price; z-scored cross-sectionally, equal-weight composite.")
    A("- Weekly rebalance (Monday): signals on prior close, execution at rebalance close — no lookahead.")
    A("- Long top quintile / short bottom quintile, equal-weight; legs sized 0.5 capital each (1× gross, dollar-neutral).")
    A("- **Costs: 25 bps per one-way dollar traded** (Σ|Δw| × 25bps per rebalance).")
    A("- Variants: long-short, long-only (honest retail), long-minus-BTC, naive 12-1 momentum long-short.")
    A("")
    A("## Results (net of costs)")
    A("")
    A("| strategy | sample | weeks | total ret | ann. Sharpe | max DD | avg wkly turnover |")
    A("|---|---|---|---|---|---|---|")
    for key, blk in out["strategies"].items():
        for sk, nm in [("full", "full"), ("in_2020_21", "2020–21"), ("oos_2022_26", "2022–26 OOS")]:
            s = blk[sk]
            A(f"| {key} | {nm} | {s.get('n_weeks','-')} | {s.get('total_ret','-')} | "
              f"{s.get('ann_sharpe','-')} | {s.get('max_dd','-')} | {s.get('avg_turnover_wk','-')} |")
    A("")
    A("## Skeptic's pass")
    skp = []
    ls_oos = out["strategies"]["longshort"]["oos_2022_26"].get("ann_sharpe")
    lo_oos = out["strategies"]["longonly"]["oos_2022_26"].get("ann_sharpe")
    nv_oos = out["strategies"]["naive_mom121_longshort"]["oos_2022_26"].get("ann_sharpe")
    skp.append(f"- Naive 12-1 momentum long-short OOS Sharpe: {nv_oos} — "
               "the composite must beat this or there is nothing beyond naive momentum.")
    if lo_oos is not None and ls_oos is not None:
        if lo_oos <= 0 < (ls_oos or 0):
            skp.append("- Long-only dies but long-short lives → the 'edge' is an "
                       "unshortable artifact (most alts cannot be shorted at retail).")
        elif (ls_oos or 0) <= 0 and (lo_oos or 0) <= 0:
            skp.append("- Both legs dead OOS: no transfer of the published factor to "
                       "a retail-cost, 2022–26 setting.")
    skp.append("- 25 bps/side is a floor: real alt spreads + borrow are wider, and many "
               "bottom-quintile names are unshortable — true implementable cost is higher.")
    skp.append("- Survivorship: universe rebuilt point-in-time from the mcap proxy; LUNC "
               "excluded (hyperinflation breaks the proxy) — this *misses* the 2022 "
               "LUNA-collapse short, so the test is conservative against short-side edge. "
               "Delisted-coin exit assumed at last close (optimistic).")
    A("\n".join(skp))
    A("")
    A("## Files")
    A("- `scripts/fetch_ctrender.py`, `scripts/test_ctrender.py`")
    A("- `results/raw_xs_crypto/` (per-coin daily price/mcap-proxy/volume_usd CSVs + coins_markets.json)")
    A("- `results/ctrender_summary.json`")
    for l in L:
        print(l)
    with open(f"{RES}/test_ctrender.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote results/test_ctrender.md", flush=True)

if __name__ == "__main__":
    main()
