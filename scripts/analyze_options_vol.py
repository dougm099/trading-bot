"""Options/volatility research: honest tests on free data.
T1: CBOE PutWrite (^PUT) vs SPY buy-hold -- edge or repackaged beta?
T2: Variance risk premium (VIX - 30d realized) as SPY timing signal.
T3: SKEW / VVIX quintiles vs forward SPY returns (descriptive).
T4: Long-vol tail hedge (SPY + VXX sleeve) -- risk-adjusted improvement?
All daily (T4 monthly), costs modeled, split-sample, no lookahead.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import yfinance as yf

OUT = {}
COST_EQ = 0.0005  # 5 bps/side equities

def dl(t: str) -> pd.Series:
    s = yf.Ticker(t).history(period="max", auto_adjust=True)["Close"]
    s.index = s.index.tz_localize(None)  # ^VIX is Chicago tz, equities NY -> normalize
    return s

def perf(r: pd.Series, label: str, ann: int = 252) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 50:
        return {"label": label, "n": n, "note": "too short"}
    tot = float((1 + r).prod() - 1)
    mu = float(r.mean() * ann)
    vol = float(r.std() * np.sqrt(ann))
    sharpe = float(mu / vol) if vol > 0 else 0.0
    cum = (1 + r).cumprod()
    dd = float(((cum / cum.cummax()) - 1).min())
    return {"label": label, "n": int(n), "total_ret": round(tot, 4),
            "ann_ret": round(mu, 4), "ann_vol": round(vol, 4),
            "sharpe_rf0": round(sharpe, 3), "maxDD": round(dd, 4)}

def half_split(s: pd.Series):
    mid = s.index[len(s) // 2]
    return s[s.index < mid], s[s.index >= mid]

def trailing_quintile(s: pd.Series, window: int = 504):
    cuts = pd.DataFrame({q: s.rolling(window, min_periods=window // 2).quantile(q)
                         for q in (0.2, 0.4, 0.6, 0.8)})
    df = pd.concat([s.rename("v"), cuts], axis=1).dropna()
    qv = (df["v"].to_numpy()[:, None] > df[[0.2, 0.4, 0.6, 0.8]].to_numpy()).sum(axis=1)
    return qv, df.index

print("downloading...")
px = {t: dl(t) for t in ["^PUT", "^VIX", "^VIX3M", "^VVIX", "^SKEW", "SPY", "VXX"]}
ret = {t: s.pct_change() for t, s in px.items()}
spy = ret["SPY"]

# ---------------- T1: CBOE PutWrite vs SPY ----------------
common = pd.concat([ret["^PUT"], spy], axis=1, join="inner").dropna()
common.columns = ["PUT", "SPY"]
d1 = {"period": f"{common.index[0].date()}..{common.index[-1].date()}",
      "PUT": perf(common["PUT"], "PUT"),
      "SPY_same_window": perf(common["SPY"], "SPY")}
b = float(np.cov(common["PUT"], common["SPY"])[0, 1] / np.var(common["SPY"]))
d1.update({"beta_to_SPY": round(b, 3),
           "alpha_ann": round(float(common["PUT"].mean() - b * common["SPY"].mean()) * 252, 4),
           "corr": round(float(common["PUT"].corr(common["SPY"])), 3)})
h1, h2 = half_split(common)
d1["first_half"] = {"PUT": perf(h1["PUT"], "PUT"), "SPY": perf(h1["SPY"], "SPY")}
d1["second_half"] = {"PUT": perf(h2["PUT"], "PUT"), "SPY": perf(h2["SPY"], "SPY")}
OUT["T1_putwrite_vs_spy"] = d1
print("T1 done")

# ---------------- T2: Variance risk premium timing ----------------
vix = px["^VIX"] / 100.0
rv30 = spy.rolling(30).std() * np.sqrt(252)
vrp = (vix - rv30).dropna()
z = ((vrp - vrp.rolling(252).mean()) / vrp.rolling(252).std()).dropna()
sig = (z > 0.5).astype(float)
mon = sig.resample("M").last().shift(1)          # decide at month-end
mon = mon.reindex(spy.index, method="ffill").fillna(0)
strat = mon * spy
turnover = mon.diff().abs().fillna(0)
strat_net = (strat - turnover * COST_EQ).dropna()
d2 = {"rule": "long SPY if trailing-1y z(VIX-30dRV)>0.5 else cash, monthly rebalance, 5bps/side",
      "signal_positive_frac": round(float(sig.mean()), 3),
      "strategy": perf(strat_net, "VRP-timing"),
      "buyhold_same_window": perf(spy.loc[strat_net.index], "SPY")}
h1, h2 = half_split(strat_net)
d2["first_half"] = {"strat": perf(h1, "s"), "bh": perf(spy.loc[h1.index], "bh")}
d2["second_half"] = {"strat": perf(h2, "s"), "bh": perf(spy.loc[h2.index], "bh")}
qv, qidx = trailing_quintile(vrp)
fwd = spy.rolling(21).sum().shift(-21)
qq = pd.DataFrame({"q": qv, "fwd": fwd.reindex(qidx).to_numpy()}, index=qidx).dropna()
d2["fwd21d_by_vrp_quintile"] = {str(k): round(float(v * 100), 3)
                                for k, v in qq.groupby("q")["fwd"].mean().items()}
d2["n_quintile"] = int(len(qq))
OUT["T2_vrp_timing"] = d2
print("T2 done")

# ---------------- T3: SKEW / VVIX quintiles ----------------
for t in ["^SKEW", "^VVIX"]:
    s = px[t].dropna()
    qv2, qi2 = trailing_quintile(s)
    d = pd.DataFrame({"q": qv2, "fwd": fwd.reindex(qi2).to_numpy()}, index=qi2).dropna()
    OUT[f"T3_{t.strip('^')}_quintiles"] = {
        "metric": "forward 21d SPY return % by trailing-2y quintile",
        "quintiles": {str(k): round(float(v * 100), 3)
                      for k, v in d.groupby("q")["fwd"].mean().items()},
        "n": int(len(d))}
print("T3 done")

# ---------------- T4: long-vol tail hedge (monthly) ----------------
al = pd.concat([spy, ret["VXX"]], axis=1, join="inner").dropna()
al.columns = ["SPY", "VXX"]
mret = al.resample("M").apply(lambda x: float((1 + x).prod() - 1))
for w_vxx, name in [(0.05, "95/5"), (0.10, "90/10")]:
    port = (1 - w_vxx) * mret["SPY"] + w_vxx * mret["VXX"]
    port.iloc[1:] = port.iloc[1:] - 2 * w_vxx * COST_EQ  # rebalance cost each month
    port = port.iloc[1:]
    OUT[f"T4_tailhedge_{name.replace('/', '_')}"] = {
        "desc": f"{name} SPY/VXX monthly-rebalanced, 5bps costs",
        "period": f"{mret.index[0].date()}..{mret.index[-1].date()}",
        "portfolio": perf(port, name, ann=12),
        "spy_same": perf(mret["SPY"].loc[port.index], "SPY", ann=12)}
print("T4 done")

with open("results/options_vol_summary.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1))
print("WROTE results/options_vol_summary.json")
