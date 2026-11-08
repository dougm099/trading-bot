#!/usr/bin/env python3
"""Index<->constituent lead-lag + sector spillover analysis (research only).

Pipeline:
  1. fetch()               - 1h bars for ~26 tickers via yfinance, cached to results/prices_1h.csv
  2. lagged_xcorr()        - cross-correlation of log returns, lags -10..+10 (positive lag = x leads y)
  3. granger_p()           - Granger causality p-values (SSR F-test) both directions
  4. granger_matrix()      - 11x11 sector spillover matrix
  5. backtest_leader_follower() - dead-simple leader->follower backtest, 5bps/side costs
No lookahead: signals only ever use data strictly before the traded bar.

Usage: python3 scripts/analyze_leadlag.py   (run from ~/workspace/trading-bot)
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
CACHE = RESULTS / "prices_1h.csv"

INDEXES = ["SPY", "QQQ", "DIA", "IWM"]
MEGA = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "AVGO", "AMD"]
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLU", "XLK"]

# (leader-ish first, follower-ish second) pairs for section 1
PAIRS = (
    [("SPY", m) for m in MEGA]
    + [("QQQ", m) for m in MEGA]
    + [("QQQ", "XLK"), ("SMH", "NVDA"), ("SMH", "AMD"), ("SMH", "AVGO")]
)

LAGS = range(-10, 11)          # hourly bars; positive = first-named leads
GRANGER_LAGS = (1, 2, 4, 8)
COST_PER_SIDE = 0.0005         # 5 bps
BARS_PER_YEAR = 252 * 7        # ~7 hourly bars per US trading day


# --------------------------------------------------------------------------- #
def fetch(tickers, period="730d", interval="1h", retries=5, use_cache=True):
    """Download 1h bars, return DataFrame of adjusted closes (UTC index, inner-joined)."""
    if use_cache and CACHE.exists():
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            return df[tickers]
    # download in chunks to dodge rate limits
    chunks, remaining = [], list(tickers)
    while remaining:
        batch, remaining = remaining[:10], remaining[10:]
        for attempt in range(retries):
            try:
                raw = yf.download(
                    batch, period=period, interval=interval,
                    auto_adjust=True, progress=False,
                    group_by="ticker" if len(batch) > 1 else None,
                )
                if len(batch) == 1:
                    close = raw[["Close"]].rename(columns={"Close": batch[0]})
                else:
                    close = raw.xs("Close", axis=1, level=1)[batch]
                chunks.append(close)
                time.sleep(2)
                break
            except Exception as e:  # noqa: BLE001 - transient network/rate-limit errors
                wait = 15 * (attempt + 1)
                print(f"  download failed ({e}), retrying in {wait}s ...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"download failed for batch {batch} after {retries} retries")
    df = pd.concat(chunks, axis=1).sort_index()
    df = df.dropna(axis=1, how="all")
    df.index = df.index.tz_convert("UTC")
    df = df[tickers]  # enforce order; raises if a ticker is entirely missing
    df.to_csv(CACHE)
    return df


def log_returns(prices):
    return np.log(prices / prices.shift(1))


# --------------------------------------------------------------------------- #
def lagged_xcorr(x, y, lags=LAGS):
    """corr(x_t, y_{t+L}) for each L. Positive L => x leads y. NaN-safe."""
    x = x - x.mean()
    y = y - y.mean()
    out = {}
    for L in lags:
        a, b = x, y.shift(-L)
        valid = a.notna() & b.notna()
        if valid.sum() < 100:
            out[L] = np.nan
            continue
        denom = a[valid].std() * b[valid].std()
        out[L] = (a[valid] * b[valid]).mean() / denom if denom > 0 else np.nan
    return out


def granger_p(y, x, lags=GRANGER_LAGS):
    """SSR F-test p-value of 'x Granger-causes y' for each lag in lags. NaN on failure."""
    data = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(data) < 50:
        return {L: np.nan for L in lags}
    res = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tests = grangercausalitytests(data[["y", "x"]], maxlag=max(lags))
        for L in lags:
            res[L] = float(tests[L][0]["ssr_ftest"][1])
    except Exception:  # noqa: BLE001 - collinearity / numerical failures
        res = {L: np.nan for L in lags}
    return res


def deseasonalize(rets):
    """Subtract the hour-of-day mean return (America/New_York) from each series.

    Descriptive only: kills the deterministic overnight-gap seasonality that
    otherwise creates spurious lag-7 cross-correlations. Not used in backtests
    (they trade raw data with past-only thresholds).
    """
    hours = rets.index.tz_convert("America/New_York").hour
    means = rets.groupby(hours).transform("mean")
    return rets - means


def granger_matrix(rets, tickers, lags=(2, 4)):
    """Ordered-pair Granger table: for each (leader, follower), p-values + xcorr summary."""
    rows = []
    for leader in tickers:
        for follower in tickers:
            if leader == follower:
                continue
            g = granger_p(rets[follower], rets[leader], lags)
            xc = lagged_xcorr(rets[leader], rets[follower])
            nz = {L: c for L, c in xc.items() if L != 0 and not np.isnan(c)}
            best_nz = max(nz, key=lambda L: abs(nz[L])) if nz else np.nan
            best = max(xc, key=lambda L: abs(xc[L])) if xc else np.nan
            vals = [g[L] for L in lags]
            minp = float(np.nanmin(vals)) if any(not np.isnan(v) for v in vals) else np.nan
            rows.append({
                "leader": leader, "follower": follower,
                **{f"granger_p_lag{L}": g[L] for L in lags},
                "min_granger_p": minp,
                "best_lag": best, "corr_at_best": xc.get(best, np.nan),
                "best_nonzero_lag": best_nz,
                "corr_at_nonzero": nz.get(best_nz, np.nan),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def backtest_leader_follower(prices, leader, follower, k, cost=COST_PER_SIDE):
    """Long/short follower for the next bar based on leader's trailing k-bar return.

    Position decided at close of bar t using leader returns through bar t and the
    trailing 60-bar std computed on bars strictly before t. Trade follower from
    close t to close t+1. 5 bps per side on every position change.
    """
    lret = np.log(prices[leader] / prices[leader].shift(1))
    fret = np.log(prices[follower] / prices[follower].shift(1))
    signal = lret.rolling(k).sum()                                  # known at close of t
    thr = 0.5 * lret.shift(1).rolling(60).std()                     # strictly past bars only
    pos = pd.Series(0, index=prices.index, dtype=float)
    pos[signal > thr] = 1.0
    pos[signal < -thr] = -1.0
    pos = pos.fillna(0.0)
    held = pos.shift(1).fillna(0.0)                                 # position held during bar t
    gross = held * fret
    churn = (pos.shift(1).fillna(0.0) - pos.shift(2).fillna(0.0)).abs()
    net = (gross - cost * churn).dropna()
    if len(net) == 0 or net.std() == 0:
        return {"k": k, "net_return": 0.0, "sharpe": 0.0, "trades": 0,
                "max_drawdown": 0.0, "win_rate": np.nan}
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    trades = int((churn > 0).sum())
    active = net[(held != 0)]
    return {
        "k": k,
        "net_return": float(np.exp(net.sum()) - 1),
        "sharpe": float(net.mean() / net.std() * np.sqrt(BARS_PER_YEAR)) if net.std() > 0 else 0.0,
        "trades": trades,
        "max_drawdown": float(dd),
        "win_rate": float((active > 0).mean()) if len(active) else np.nan,
    }


# --------------------------------------------------------------------------- #
def main():
    RESULTS.mkdir(exist_ok=True)
    tickers = sorted(set(INDEXES + MEGA + ["SMH"] + SECTORS))
    print(f"Fetching {len(tickers)} tickers, 1h bars, ~2y ...")
    prices = fetch(tickers)
    prices = prices.dropna(axis=0, how="any")   # common timestamps only
    print(f"  rows={len(prices)} from {prices.index[0]} to {prices.index[-1]}")
    rets = log_returns(prices)
    # deseasonalized returns for measurement only (kills spurious lag-7 daily seasonality)
    mrets = deseasonalize(rets)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("SPY autocorr lag-7 raw vs deseasonalized:",
              round(rets["SPY"].corr(rets["SPY"].shift(7)), 4), "/",
              round(mrets["SPY"].corr(mrets["SPY"].shift(7)), 4))

    # ---- 1. index<->constituent pairs --------------------------------------
    print("\n== Section 1: index<->constituent lead-lag ==")
    pair_rows = []
    for a, b in PAIRS:
        ra, rb = mrets[a], mrets[b]
        xc = lagged_xcorr(ra, rb)
        best = max(xc, key=lambda L: abs(xc[L]))
        nz = {L: c for L, c in xc.items() if L != 0}
        best_nz = max(nz, key=lambda L: abs(nz[L]))
        g_ab = granger_p(rb, ra)   # a -> b
        g_ba = granger_p(ra, rb)   # b -> a
        row = {"pair": f"{a}-{b}", "a": a, "b": b,
               "best_lag": best, "corr_at_best": xc[best],
               "best_nonzero_lag": best_nz, "corr_at_nonzero": nz[best_nz]}
        for L in GRANGER_LAGS:
            row[f"granger_{a}_to_{b}_p{L}"] = g_ab[L]
            row[f"granger_{b}_to_{a}_p{L}"] = g_ba[L]
        pair_rows.append(row)
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(RESULTS / "leadlag_table.csv", index=False)
    cols = ["pair", "best_lag", "corr_at_best", "best_nonzero_lag", "corr_at_nonzero"]
    gcols = [c for c in pairs_df.columns if c.startswith("granger_")]
    print(pairs_df[cols + [c for c in gcols if "_p4" in c]].to_string(index=False))

    # ---- 2. sector spillover matrix -----------------------------------------
    print("\n== Section 2: 11x11 sector spillover ==")
    sec = granger_matrix(mrets, SECTORS)
    sec.to_csv(RESULTS / "sector_spillover.csv", index=False)
    ranked = sec.sort_values("min_granger_p").head(15)
    print(ranked[["leader", "follower", "granger_p_lag2", "granger_p_lag4",
                  "best_nonzero_lag", "corr_at_nonzero"]].to_string(index=False))
    # narrative sanity checks
    print("\n-- narrative checks (leader -> follower) --")
    for lead, fol in [("XLE", "XLI"), ("XLK", "XLC"), ("XLF", "XLRE")]:
        r = sec[(sec.leader == lead) & (sec.follower == fol)].iloc[0]
        print(f"{lead}->{fol}: best_nonzero_lag={r['best_nonzero_lag']}, "
              f"corr={r['corr_at_nonzero']:.4f}, granger_p2={r['granger_p_lag2']:.3g}, "
              f"granger_p4={r['granger_p_lag4']:.3g}")

    # ---- 3. backtests for every candidate -----------------------------------
    print("\n== Section 3: leader->follower backtests (5bps/side) ==")
    cand = []
    # from section-1 pairs, both directions
    for _, r in pairs_df.iterrows():
        a, b = r["a"], r["b"]
        for lead, fol in [(a, b), (b, a)]:
            g = min(r[f"granger_{lead}_to_{fol}_p{L}"] for L in GRANGER_LAGS)
            xc = lagged_xcorr(mrets[lead], mrets[fol])
            nz = {L: c for L, c in xc.items() if L != 0 and not np.isnan(c)}
            c_nz = max(abs(c) for c in nz.values()) if nz else 0.0
            L_nz = max(nz, key=lambda L: abs(nz[L])) if nz else np.nan
            if c_nz > 0.05 or (not np.isnan(g) and g < 0.05):
                cand.append((lead, fol, g, L_nz, nz.get(L_nz, np.nan), "pair"))
    # from sector matrix
    for _, r in sec.iterrows():
        if abs(r["corr_at_nonzero"]) > 0.05 or r["min_granger_p"] < 0.05:
            cand.append((r["leader"], r["follower"], r["min_granger_p"],
                         r["best_nonzero_lag"], r["corr_at_nonzero"], "sector"))
    cand = list({(c[0], c[1]): c for c in cand}.values())  # dedupe
    print(f"  {len(cand)} candidates pass the screen")
    bt_rows = []
    for lead, fol, g, L_nz, c_nz, src in cand:
        for k in (1, 4):
            m = backtest_leader_follower(prices, lead, fol, k)
            bt_rows.append({"leader": lead, "follower": fol, "src": src,
                            "granger_p": g, "best_nonzero_lag": L_nz,
                            "corr_at_nonzero": c_nz, **m})
    bt = pd.DataFrame(bt_rows)
    bt.to_csv(RESULTS / "leadlag_backtests.csv", index=False)
    top = bt.sort_values("sharpe", ascending=False).head(10)
    print(top[["leader", "follower", "k", "granger_p", "best_nonzero_lag",
               "corr_at_nonzero", "net_return", "sharpe", "trades",
               "max_drawdown"]].to_string(index=False))
    print("\nCSVs saved to", RESULTS)


if __name__ == "__main__":
    main()
