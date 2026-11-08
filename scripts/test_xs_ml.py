#!/usr/bin/env python3
"""T4: Cross-sectional ML ranking on US equities (Gu-Kelly-Xiu style, price-only).

Mechanism: predict RELATIVE next-month returns across hundreds of stocks (not
single-asset direction). Universe: broad US common stocks (Nasdaq Trader list),
eligible set recomputed monthly (price>=5, dollar-volume filter). Features:
momentum 1m/6m/12m, 12m realized vol, dollar volume, 52wk-high proximity,
log price, short-term reversal. Monthly formation, expanding-window refit
annually. Models: OLS, PCR, HistGradientBoosting vs dumb momentum-sort baseline.
Portfolios: long-short decile, long-only top decile, top-decile minus SPY.
Costs: 40 bps round-trip on actual turnover. Kill: net Sharpe < SPY buy-hold,
or DSR failure. Regime splits incl. 2008/2020/2022.

Survivorship: universe list is 2026-vintage -> survivors favored; disclosed.
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "results", "raw_xs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
COST_RT = 0.004  # 40 bps round-trip
FIRST_OOS_YEAR = 2005

def load_panel():
    p = pd.read_parquet(os.path.join(RAW, "universe.parquet"))
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    return p

def add_features(p):
    g = p.groupby("ticker", sort=False)
    ac = p["adjclose"]; cl = p["close"]; hi = p["high"]; vo = p["volume"]
    ret1 = g[ac.name].pct_change() if False else g["adjclose"].pct_change()
    p["ret1"] = ret1.values
    p["mom1"]  = g["adjclose"].pct_change(1).shift(1).values          # prior-month (reversal)
    p["mom6"]  = (g["adjclose"].shift(2) / g["adjclose"].shift(8) - 1).values
    p["mom12"] = (g["adjclose"].shift(2) / g["adjclose"].shift(14) - 1).values
    p["vol12"] = (g["ret1"].rolling(12).std().values * math.sqrt(12))
    p["dolvol"] = np.log((p["volume"] * p["close"]).replace(0, np.nan)
                         .groupby(p["ticker"]).rolling(12).mean().values)
    rollmax = g["high"].rolling(12).max().shift(1).values
    p["high52"] = (g["close"].shift(1).values / rollmax)
    p["logpx"] = np.log(g["close"].shift(1).values)
    p["label"] = g["ret1"].shift(-1).values   # month t+1 return, formed at end of t
    return p

FEATS = ["mom1", "mom6", "mom12", "vol12", "dolvol", "high52", "logpx"]

def eligible(df):
    # at formation month t: price>=5, dollar volume filter, features present
    px = df.groupby("ticker")["close"].shift(1)
    dv = (df["volume"] * df["close"]).groupby(df["ticker"]).rolling(12).mean().values
    ok = (px.values >= 5) & (dv >= 20e6)  # ~$1M/day
    ok = ok & df[FEATS].notna().all(axis=1).values
    return ok

def zscore_monthly(df):
    df = df.copy()
    for f in FEATS:
        mu = df.groupby("date")[f].transform("mean")
        sd = df.groupby("date")[f].transform("std").replace(0, np.nan)
        df[f] = ((df[f] - mu) / sd).clip(-3, 3)
    return df

def fit_predict(train, test_months, method):
    """train: DataFrame of stock-months; test_months: sorted list of dates.
    Returns DataFrame with columns date,ticker,pred."""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xtr = train[FEATS].values; ytr = train["label"].values
    mask = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
    Xtr, ytr = Xtr[mask], ytr[mask]
    if method == "ols":
        mdl = LinearRegression()
    elif method == "pcr":
        mdl = (PCA(n_components=6, random_state=42), Ridge(alpha=1.0))
    elif method == "hgb":
        mdl = HistGradientBoostingRegressor(max_iter=100, max_depth=3,
                                            learning_rate=0.05, min_samples_leaf=50,
                                            random_state=42)
    elif method == "momsort":
        mdl = None
    else:
        raise ValueError(method)
    if mdl is not None:
        if method == "pcr":
            pca, ridge = mdl
            Z = pca.fit_transform(Xtr); ridge.fit(Z, ytr)
        else:
            mdl.fit(Xtr, ytr)
    outs = []
    for d in test_months:
        sub = train_test_lookup[d]  # set by caller via global
        X = sub[FEATS].values
        if method == "momsort":
            pred = sub["mom12"].values  # dumb baseline: classic 12-2 momentum rank
        elif method == "pcr":
            pred = ridge.predict(pca.transform(X))
        else:
            pred = mdl.predict(X)
        outs.append(pd.DataFrame({"date": d, "ticker": sub["ticker"].values, "pred": pred}))
    return pd.concat(outs, ignore_index=True)

def decile_portfolios(preds, rets):
    """preds: date,ticker,pred. rets: date,ticker,ret (next-month return).
    Returns monthly series dict."""
    m = preds.merge(rets, on=["date", "ticker"], how="inner")
    rows = []
    prev_long, prev_short = set(), set()
    for d, g in m.groupby("date", sort=True):
        g = g.dropna(subset=["pred", "ret"])
        n = len(g)
        if n < 50: continue
        g = g.assign(dec=pd.qcut(g["pred"], 10, labels=False, duplicates="drop"))
        top = set(g.loc[g["dec"] == g["dec"].max(), "ticker"])
        bot = set(g.loc[g["dec"] == g["dec"].min(), "ticker"])
        rt, rb = g.loc[g["ticker"].isin(top), "ret"].mean(), g.loc[g["ticker"].isin(bot), "ret"].mean()
        to_l = 1.0 if not prev_long else len(top - prev_long) / max(len(top), 1)
        to_s = 1.0 if not prev_short else len(bot - prev_short) / max(len(bot), 1)
        prev_long, prev_short = top, bot
        rows.append({"date": d, "n": n,
                     "ls_gross": 0.5 * (rt - rb), "to_ls": 0.5 * (to_l + to_s),
                     "long": rt, "to_long": to_l})
    return pd.DataFrame(rows).set_index("date").sort_index()

def sharpe(s):
    s = s.dropna()
    if len(s) < 12 or s.std() == 0: return np.nan
    return s.mean() / s.std() * math.sqrt(12)

def maxdd(s):
    c = (1 + s.fillna(0)).cumprod()
    return ((c / c.cummax()) - 1).min()

def dsr(sr_hat, trial_sharpes, rets):
    """Deflated Sharpe Ratio (Bailey & LdP 2014). Returns (DSR, expected null Sharpe)."""
    import scipy.stats as st
    S = len(trial_sharpes)
    V = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    g = 0.5772156649
    sr0 = math.sqrt(max(V, 1e-12)) * ((1 - g) * st.norm.ppf(1 - 1 / S)
                                       + g * st.norm.ppf(1 - 1 / (S * math.e)))
    T = len(rets)
    rs = pd.Series(np.asarray(rets, dtype=float)).dropna()
    skew, ekurt = float(rs.skew()), float(rs.kurtosis())  # kurtosis() = excess
    num = (sr_hat - sr0) * math.sqrt(max(T - 1, 1))
    den = math.sqrt(max(1 - skew * sr_hat + (ekurt / 4) * sr_hat ** 2, 1e-9))
    return float(st.norm.cdf(num / den)), float(sr0)

def main():
    t0 = time.time()
    p = load_panel()
    print(f"panel: {len(p)} rows, {p['ticker'].nunique()} tickers", flush=True)
    p = add_features(p)
    p["elig"] = eligible(p)
    d = p[p["elig"]].copy()
    d = zscore_monthly(d)
    # keep only formations with a label available
    d = d[d["label"].notna()].copy()
    print(f"eligible stock-months: {len(d)}, months: {d['date'].nunique()}, "
          f"range {d['date'].min().date()}..{d['date'].max().date()}", flush=True)

    rets = d[["date", "ticker", "label"]].rename(columns={"label": "ret"})
    global train_test_lookup
    train_test_lookup = {dt: g for dt, g in d.groupby("date", sort=True)}

    years = sorted(d["date"].dt.year.unique())
    oos_years = [y for y in years if y >= FIRST_OOS_YEAR]
    methods = ["ols", "pcr", "hgb", "momsort"]
    all_preds = {m: [] for m in methods}
    oos_r2 = {m: {"num0": 0.0, "den0": 0.0, "numm": 0.0, "denm": 0.0} for m in methods}

    for Y in oos_years:
        tr = d[d["date"] < f"{Y}-01-01"]
        tmonths = sorted(d[(d["date"].dt.year == Y)]["date"].unique())
        if len(tr) == 0 or len(tmonths) == 0: continue
        print(f"[{Y}] train stock-months={len(tr)}, test months={len(tmonths)}", flush=True)
        for m in methods:
            pr = fit_predict(tr, tmonths, m)
            all_preds[m].append(pr)
            # OOS R^2 accumulators (vs 0 and vs monthly mean)
            mm = pr.merge(rets, on=["date", "ticker"])
            for dt, g in mm.groupby("date"):
                r = g["ret"].values; f = g["pred"].values
                okm = np.isfinite(r) & np.isfinite(f)
                r, f = r[okm], f[okm]
                if len(r) < 20: continue
                o = oos_r2[m]
                o["num0"] += np.sum((r - f) ** 2); o["den0"] += np.sum(r ** 2)
                mu = r.mean()
                o["numm"] += np.sum((r - f) ** 2); o["denm"] += np.sum((r - mu) ** 2)
    print(f"modeling done in {(time.time()-t0)/60:.1f} min", flush=True)

    # SPY benchmark
    import yfinance as yf
    raw = yf.download("SPY", period="max", interval="1mo", auto_adjust=False,
                      progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        _ac = raw["Adj Close"]
        spy = _ac["SPY"] if "SPY" in _ac.columns else _ac.iloc[:, 0]
    else:
        spy = raw["Adj Close"]
    spy = pd.Series(np.asarray(spy).ravel(), index=raw.index)
    if getattr(spy.index, "tz", None):
        spy.index = spy.index.tz_localize(None)
    spy_m = spy.resample("ME").last().pct_change()

    results = {}
    for m in methods:
        preds = pd.concat(all_preds[m], ignore_index=True)
        pf = decile_portfolios(preds, rets)
        pf["ls_net"] = pf["ls_gross"] - COST_RT * pf["to_ls"]
        pf["long_net"] = pf["long"] - COST_RT * pf["to_long"]
        sm = spy_m.reindex(pf.index, method="nearest", tolerance=pd.Timedelta("40d")).fillna(0)
        pf["hedged_net"] = pf["long_net"] - sm.values  # long top decile, short SPY
        pf["spy"] = sm.values
        o = oos_r2[m]
        r2_0 = 1 - o["num0"] / o["den0"] if o["den0"] else np.nan
        r2_m = 1 - o["numm"] / o["denm"] if o["denm"] else np.nan
        # regime Sharpe (net LS and net long)
        pf["yr"] = pf.index.year
        def reg(sh):
            out = {}
            for name, cond in [("2005-07", pf["yr"].between(2005, 2007)),
                               ("2008-09", pf["yr"].between(2008, 2009)),
                               ("2010-19", pf["yr"].between(2010, 2019)),
                               ("2020", pf["yr"] == 2020), ("2021", pf["yr"] == 2021),
                               ("2022", pf["yr"] == 2022), ("2023-26", pf["yr"] >= 2023)]:
                out[name] = round(sharpe(pf.loc[cond, sh]), 3)
            return out
        res = {
            "r2_vs_zero": round(r2_0, 4), "r2_vs_mean": round(r2_m, 4),
            "n_months": len(pf), "avg_stocks": round(pf["n"].mean(), 1),
            "ls_gross_sharpe": round(sharpe(pf["ls_gross"]), 3),
            "ls_net_sharpe": round(sharpe(pf["ls_net"]), 3),
            "ls_net_annret": round(pf["ls_net"].mean() * 12, 4),
            "ls_net_maxdd": round(maxdd(pf["ls_net"]), 4),
            "long_net_sharpe": round(sharpe(pf["long_net"]), 3),
            "long_net_annret": round(pf["long_net"].mean() * 12, 4),
            "hedged_net_sharpe": round(sharpe(pf["hedged_net"]), 3),
            "spy_sharpe": round(sharpe(pf["spy"]), 3),
            "spy_annret": round(pf["spy"].mean() * 12, 4),
            "regime_ls_net_sharpe": reg("ls_net"),
            "regime_long_net_sharpe": reg("long_net"),
            "avg_monthly_turnover_ls": round(pf["to_ls"].mean(), 3),
        }
        results[m] = res
        print(json.dumps({m: res}, indent=1), flush=True)
        pf.to_csv(os.path.join(RES, f"xs_portfolio_{m}.csv"))

    # DSR on best net LS Sharpe; trials = methods x portfolios (ls_net, long_net, hedged_net)
    trial_sharpes = [results[m][k] for m in methods
                     for k in ("ls_net_sharpe", "long_net_sharpe", "hedged_net_sharpe")]
    trials = len(trial_sharpes)
    best_m = max(methods, key=lambda m: results[m]["ls_net_sharpe"])
    sr_hat = results[best_m]["ls_net_sharpe"]
    pf_best = pd.read_csv(os.path.join(RES, f"xs_portfolio_{best_m}.csv"),
                          index_col=0, parse_dates=True)["ls_net"].dropna()
    p_dsr, e0 = dsr(sr_hat, trial_sharpes, pf_best)
    results["_dsr"] = {"best_method": best_m, "trials": trials,
                       "dsr": round(p_dsr, 4), "expected_null_sharpe": round(e0, 3),
                       "spy_benchmark_sharpe": results[best_m]["spy_sharpe"]}
    with open(os.path.join(RES, "xs_ml_summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("DSR:", results["_dsr"], flush=True)
    print(f"TOTAL {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
