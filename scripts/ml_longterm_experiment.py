"""Long-horizon cross-sectional ML experiment. See results/ml_longterm_RULEBOOK.md
(pre-declared 2026-09-20 before any data was seen).

Pipeline: month-end signal dates -> 10 point-in-time features -> forward
tercile labels -> walk-forward (expanding, 64-trading-day embargo) ->
4 pre-declared model configs -> overlapping 63d cohort portfolios + EW benchmark.
"""
from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats as sps

DATA = os.path.expanduser("~/workspace/trading-bot/results/datasets/longterm")
RES = os.path.expanduser("~/workspace/trading-bot/results")
HORIZONS = [21, 42, 63]
PRIMARY_H = 63
COST = 0.001  # 10 bps per side
TOP_N = 25
TEST_YEARS = list(range(2016, 2026))

SECTOR_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Energy": "XLE",
    "Healthcare": "XLV", "Industrials": "XLI", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Utilities": "XLU", "Real Estate": "XLRE",
    "Basic Materials": "XLB", "Communication Services": "XLC",
}

CONFIGS = [
    ("hgb_d3", HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                             learning_rate=0.05, random_state=0)),
    ("hgb_d5", HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                                             learning_rate=0.05, random_state=0)),
    ("logit_c1", LogisticRegression(C=1.0, max_iter=2000)),
    ("logit_c01", LogisticRegression(C=0.1, max_iter=2000)),
]
PRIMARY_CFG = "hgb_d3"

FEATS = ["mom_12_1", "mom_6", "rev_1m", "dist_52w_high", "vol_63", "dd_252",
         "turnover_trend", "sector_rel_mom", "mkt_trend", "vix_level"]


def load():
    bars = pd.read_parquet(f"{DATA}/bars.parquet")
    closes = bars["Close"].astype(float)
    highs = bars["High"].astype(float)
    volumes = bars["Volume"].astype(float)
    # ticker -> sector ETF (from Wikipedia GICS, cached by download script)
    etfmap = pd.read_csv(f"{DATA}/sector_etf.csv").set_index("ticker")["etf"].to_dict()
    return closes, highs, volumes, etfmap


def signal_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = idx.to_series()
    return pd.DatetimeIndex(sorted(s.groupby([s.dt.year, s.dt.month]).max().tolist()))


def build_samples(closes, highs, volumes, sectors, horizon):
    """One row per (signal date, ticker): 10 features + forward return."""
    idx = closes.index
    sdates = signal_dates(idx)
    tickers = [t for t in closes.columns if t not in
               ("SPY", "^VIX", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
                "XLY", "XLU", "XLRE", "XLB", "XLC")]
    pos_of = {d: i for i, d in enumerate(idx)}
    spos = np.array([pos_of[d] for d in sdates])

    # market series (positional, full history assumed for SPY/^VIX)
    spy = closes["SPY"].to_numpy()
    vix_raw = closes["^VIX"].to_numpy()
    mkt_trend = spy[spos] / spy[spos - 63] - 1
    # trailing 5y z-score of VIX level (point-in-time)
    vix_z = np.full(len(sdates), np.nan)
    for j, p in enumerate(spos):
        w = vix_raw[max(0, p - 1260):p + 1]
        w = w[~np.isnan(w)]
        if len(w) > 100 and np.std(w) > 1e-9:
            vix_z[j] = (vix_raw[p] - w.mean()) / w.std()
    # sector ETF 12-1 momentum per signal date
    sec_mom = {}
    for etf in sorted(set(SECTOR_ETF.values())):
        e = closes[etf].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            sec_mom[etf] = e[spos - 21] / e[spos - 252] - 1

    rows = []
    mom12_store = {}  # (ticker) -> dict date->mom12_1, for universe-mean fallback
    for t in tickers:
        c = closes[t].dropna()
        if len(c) < 400:
            continue
        cv, hv, vv = c.to_numpy(), highs[t].reindex(c.index).to_numpy(), volumes[t].reindex(c.index).to_numpy()
        hmax = pd.Series(hv).rolling(252, min_periods=252).max().to_numpy()
        lr = np.log(cv)
        vol63 = pd.Series(lr).rolling(63, min_periods=63).std().to_numpy() * math.sqrt(252)
        p = c.index.get_indexer(sdates)
        ok = (p >= 252) & (p + 1 + horizon < len(c))
        for j in np.where(ok)[0]:
            i = p[j]
            if not np.isfinite([cv[i], cv[i - 21], cv[i - 252], cv[i - 126], cv[i - 63],
                                cv[i + 1], cv[i + 1 + horizon], hmax[i], vol63[i]]).all():
                continue
            if cv[i - 252] == 0 or hmax[i] == 0:
                continue
            v21 = np.nanmean(vv[i - 20:i + 1]); v63 = np.nanmean(vv[i - 62:i + 1])
            if not np.isfinite(v21) or not np.isfinite(v63) or v63 <= 0:
                continue
            w = cv[i - 252:i + 1]
            dd = float(np.min(w / np.maximum.accumulate(w) - 1))
            fwd = cv[i + 1 + horizon] / cv[i + 1] - 1
            mom12_1 = cv[i - 21] / cv[i - 252] - 1
            etf = sectors.get(t, "")
            sm = sec_mom.get(etf)
            smv = sm[j] if sm is not None else np.nan
            srel = mom12_1 - smv if np.isfinite(smv) else np.nan
            rows.append((sdates[j], t, mom12_1, cv[i] / cv[i - 126] - 1,
                         cv[i] / cv[i - 21] - 1, cv[i] / hmax[i] - 1, vol63[i],
                         dd, v21 / v63, srel, mkt_trend[j], vix_z[j], fwd))
            mom12_store.setdefault(t, {})[sdates[j]] = mom12_1

    cols = ["date", "ticker"] + FEATS + ["fwd_ret"]
    df = pd.DataFrame(rows, columns=cols)
    # sector fallback: universe-mean-demeaned momentum where sector unknown
    mom12 = pd.DataFrame(
        {t: pd.Series(d) for t, d in mom12_store.items()}).sort_index()
    umean = mom12.mean(axis=1)
    mask = df["sector_rel_mom"].isna()
    df.loc[mask, "sector_rel_mom"] = (
        df.loc[mask, "mom_12_1"] - df.loc[mask, "date"].map(umean))
    # labels: top-tercile of forward return per date
    df["label"] = 0
    for d, g in df.groupby("date"):
        if len(g) >= 30:
            cut = g["fwd_ret"].quantile(2 / 3)
            df.loc[g.index, "label"] = (g["fwd_ret"] >= cut).astype(int)
    # cross-sectional z-score of stock-specific features per date
    stock_feats = [f for f in FEATS if f not in ("mkt_trend", "vix_level")]
    for d, g in df.groupby("date"):
        mu, sd = g[stock_feats].mean(), g[stock_feats].std().clip(lower=1e-8)
        df.loc[g.index, stock_feats] = (g[stock_feats] - mu) / sd
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    return df, sdates


def embargo_cutoff(idx, first_test_date, horizon):
    p = idx.get_loc(first_test_date)
    return idx[p - (horizon + 1)]


def walk_forward(df, idx, horizon):
    """Fit 4 configs per test year; return frame with proba columns."""
    df = df.copy()
    for name, _ in CONFIGS:
        df[f"p_{name}"] = np.nan
    for Y in TEST_YEARS:
        test_dates = sorted(d for d in df["date"].unique() if d.year == Y)
        if not test_dates:
            continue
        cutoff = embargo_cutoff(idx, test_dates[0], horizon)
        # leakage audit: no train label window may touch the test block
        assert idx[idx.get_loc(cutoff) + horizon + 1] <= test_dates[0]
        train = df[df["date"] <= cutoff]
        test = df[df["date"].isin(test_dates)]
        Xtr, ytr = train[FEATS].to_numpy(), train["label"].to_numpy()
        Xte = test[FEATS].to_numpy()
        for name, mdl in CONFIGS:
            mdl.fit(Xtr, ytr)
            df.loc[test.index, f"p_{name}"] = mdl.predict_proba(Xte)[:, 1]
    return df


def simulate(df, idx, closes, horizon, cfg_name):
    """Monthly rebalance: top-25 predicted names, equal weight.
    Entry at next close after signal; turnover vs drifted weights;
    10bps per side. Returns (equity Series, cohorts DataFrame)."""
    pcol = f"p_{cfg_name}"
    test = df.dropna(subset=[pcol]).copy()
    rets = closes.pct_change()
    val, vals = 1.0, {}
    w = None
    names_prev, prev_entry = [], None
    test_dates = sorted(test["date"].unique())
    cohort_rows = []
    for k, d in enumerate(test_dates):
        g = test[test["date"] == d].sort_values(pcol, ascending=False).head(TOP_N)
        entry = idx[idx.get_loc(d) + 1]
        names = g["ticker"].tolist()
        tgt = pd.Series(1.0 / len(names), index=names)
        if w is not None:
            r_prev = (closes.loc[entry, names_prev] / closes.loc[prev_entry, names_prev] - 1).fillna(0)
            drifted = w * (1 + r_prev)
            drifted = drifted / drifted.sum()
            turnover = 0.5 * (tgt.sub(drifted, fill_value=0)).abs().sum()
        else:
            turnover = 1.0
        cost = turnover * COST
        val *= (1 - cost)
        nxt_entry = idx[idx.get_loc(test_dates[k + 1]) + 1] if k < len(test_dates) - 1 else idx[-1]
        mret = rets.loc[entry:nxt_entry, names].mean(axis=1, skipna=True).fillna(0)
        g_ret = float(np.prod(1 + mret) - 1)
        for day, r in mret.items():
            val *= (1 + r)
            vals[day] = val
        cohort_rows.append({"date": d, "net_ret": (1 + g_ret) * (1 - cost) - 1,
                            "n": len(names), "turnover": float(turnover),
                            "cost": float(cost)})
        w, names_prev, prev_entry = tgt, names, entry
    eq = pd.Series(vals).sort_index()
    return eq / eq.iloc[0], pd.DataFrame(cohort_rows)


def benchmark(idx, closes, test_dates):
    """Equal-weight universe, monthly rebalance, 10bps/side on turnover."""
    excl = ("SPY", "^VIX", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
            "XLY", "XLU", "XLRE", "XLB", "XLC")
    tickers = [t for t in closes.columns if t not in excl]
    rets = closes[tickers].pct_change()
    val = 1.0
    vals = {}
    w = None
    names_prev, prev_entry = [], None
    for k, d in enumerate(test_dates):
        entry = idx[idx.get_loc(d) + 1]
        names = [t for t in tickers if closes[t].loc[:entry].notna().sum() > 252]
        tgt = pd.Series(1.0 / len(names), index=names)
        if w is not None:
            r_prev = (closes.loc[entry, names_prev] / closes.loc[prev_entry, names_prev] - 1).fillna(0)
            drifted = (w * (1 + r_prev))
            drifted = drifted / drifted.sum()
            turnover = 0.5 * (tgt.sub(drifted, fill_value=0)).abs().sum()
        else:
            turnover = 1.0
        val *= (1 - turnover * COST)
        nxt_entry = idx[idx.get_loc(test_dates[k + 1]) + 1] if k < len(test_dates) - 1 else idx[-1]
        mret = rets.loc[entry:nxt_entry, names].mean(axis=1, skipna=True).fillna(0)
        for day, r in mret.items():
            val *= (1 + r)
            vals[day] = val
        w, names_prev, prev_entry = tgt, names, entry
    eq = pd.Series(vals).sort_index()
    return eq / eq.iloc[0]


def holm_adjust(ps):
    """Holm step-down adjusted p-values (no statsmodels dependency)."""
    ps = np.asarray(ps, dtype=float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(running, 1.0)
    return adj


def perf(equity):
    r = equity.pct_change().dropna()
    n = len(r)
    sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / n) - 1 if n else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    return {"sharpe": sharpe, "cagr": cagr, "maxdd": dd, "n_days": n,
            "skew": float(r.skew()), "kurt": float(r.kurtosis() + 3)}


def deflated_sharpe(sr_d, T, skew, kurt, trial_srs):
    """Bailey & Lopez de Prado (2014), SR in daily units."""
    K = len(trial_srs)
    V = float(np.var(trial_srs, ddof=1)) if K > 1 else 0.0
    if V <= 0:
        return float("nan")
    gamma = 0.5772156649
    sr0 = math.sqrt(V) * ((1 - gamma) * sps.norm.ppf(1 - 1 / K)
                          + gamma * sps.norm.ppf(1 - 1 / (K * math.e)))
    num = (sr_d - sr0) * math.sqrt(T - 1)
    den = math.sqrt(max(1e-12, 1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2))
    return float(sps.norm.cdf(num / den))


def main():
    closes, highs, volumes, sectors = load()
    idx = closes.index
    out_rows = []
    for horizon in HORIZONS:
        print(f"=== horizon {horizon}d ===", flush=True)
        df, sdates = build_samples(closes, highs, volumes, sectors, horizon)
        print(f"samples: {len(df)}, dates: {df['date'].nunique()}, "
              f"tickers: {df['ticker'].nunique()}", flush=True)
        df.to_parquet(f"{RES}/datasets/longterm/samples_{horizon}d.parquet")
        df = walk_forward(df, idx, horizon)
        # accuracy vs baseline per config (diagnostic)
        te = df.dropna(subset=["p_hgb_d3"])
        for name, _ in CONFIGS:
            pred = (te[f"p_{name}"] >= 0.5).astype(int)
            acc = (pred == te["label"]).mean()
            base = max(te["label"].mean(), 1 - te["label"].mean())
            print(f"  {name}: acc={acc:.3f} baseline={base:.3f}", flush=True)
        # portfolios for all configs at 63d; primary cfg at 21/42d
        cfgs = [c for c, _ in CONFIGS] if horizon == PRIMARY_H else [PRIMARY_CFG]
        for cfg in cfgs:
            eq, coh = simulate(df, idx, closes, horizon, cfg)
            m = perf(eq)
            m.update({"horizon": horizon, "config": cfg,
                      "n_cohorts": len(coh),
                      "hit_rate": float((coh["net_ret"] > 0).mean()) if len(coh) else 0.0,
                      "avg_names": float(coh["n"].mean()) if len(coh) else 0.0,
                      "avg_turnover": float(coh["turnover"].mean()) if len(coh) else 0.0,
                      "total_cost": float(coh["cost"].sum()) if len(coh) else 0.0})
            out_rows.append(m)
            print(f"  [{cfg}] sharpe={m['sharpe']:.2f} cagr={m['cagr']:.3f} "
                  f"maxdd={m['maxdd']:.3f} hit={m['hit_rate']:.2f} n={m['n_cohorts']}", flush=True)
            if horizon == PRIMARY_H and cfg == PRIMARY_CFG:
                eq.to_csv(f"{RES}/datasets/longterm/equity_primary_63d.csv")
                coh.to_csv(f"{RES}/datasets/longterm/cohorts_primary_63d.csv", index=False)
    # benchmark (primary horizon test dates)
    df63 = pd.read_parquet(f"{RES}/datasets/longterm/samples_63d.parquet")
    tdates = sorted(d for d in df63["date"].unique() if d.year in TEST_YEARS)
    beq = benchmark(idx, closes, tdates)
    bm = perf(beq)
    print(f"benchmark: sharpe={bm['sharpe']:.2f} cagr={bm['cagr']:.3f} maxdd={bm['maxdd']:.3f}", flush=True)
    beq.to_csv(f"{RES}/datasets/longterm/equity_benchmark.csv")

    # DSR on primary horizon across the 4 configs (daily units)
    prim = [r for r in out_rows if r["horizon"] == PRIMARY_H]
    trial_daily = np.array([r["sharpe"] / math.sqrt(252) for r in prim])
    d = next(r for r in prim if r["config"] == PRIMARY_CFG)
    dsr = deflated_sharpe(d["sharpe"] / math.sqrt(252), d["n_days"], d["skew"],
                          d["kurt"], trial_daily)
    # Holm on one-sided Sharpe t-tests
    ps = []
    for r in prim:
        t = r["sharpe"] / math.sqrt(252) * math.sqrt(r["n_days"])
        ps.append(float(sps.t.sf(t, r["n_days"] - 1)))
    holm = holm_adjust(ps)
    print(f"DSR(primary 63d, K=4) = {dsr:.3f}; Holm adj p = {dict(zip([r['config'] for r in prim], holm))}", flush=True)

    pd.DataFrame(out_rows).to_csv(f"{RES}/ml_longterm_results.csv", index=False)
    with open(f"{RES}/ml_longterm_metrics.txt", "w") as f:
        f.write(f"benchmark_sharpe={bm['sharpe']:.4f} benchmark_cagr={bm['cagr']:.4f} "
                f"benchmark_maxdd={bm['maxdd']:.4f}\n")
        f.write(f"dsr_primary_63d_K4={dsr:.4f}\n")
        for r, p in zip(prim, holm):
            f.write(f"holm_{r['config']}={p:.4f}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
