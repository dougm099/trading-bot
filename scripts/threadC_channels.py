"""Thread C: oil->markets channel diagnostics (DIAGNOSTIC ONLY, not backtests).

Log returns; the 2 negative-price days (Apr 20-21 2020) become NaN and are
dropped from pairwise stats. All windows trailing => point-in-time.
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = os.path.expanduser("~/workspace/trading-bot/results/datasets/threadC_daily.csv")
px = pd.read_csv(DATA, index_col=0, parse_dates=True)
lr = np.log(px / px.shift(1))
vol60 = lr.rolling(60).std()
oil = "CL=F"
O, U, X, S, D = lr[oil], lr["USO"], lr["XLE"], lr["SPY"], lr["UUP"]


def cov(a, b):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    return np.cov(d["a"], d["b"])[0, 1]


def var(a):
    return a.dropna().var()


print("== contemporaneous corr (daily log ret) ==")
print(lr[[oil, "USO", "XLE", "SPY", "UUP"]].corr().round(3).to_string())

print("\n== betas to oil (full-sample realized) ==")
print(f"OIL->XLE beta: {cov(O, X)/var(O):.2f}")
print(f"OIL->SPY beta: {cov(O, S)/var(O):.2f}")
print(f"OIL->UUP beta: {cov(O, D)/var(O):.2f}")

print("\n== lead-lag: corr(oil_ret_t, asset_ret_t+1) ==")
for name, s in [("XLE", X), ("SPY", S)]:
    d = pd.DataFrame({"o": O, "f": s.shift(-1)}).dropna()
    print(f"oil(t) -> {name}(t+1): {d['o'].corr(d['f']):+.3f}")

print("\n== oil SHOCK days: |oil log ret| > 2*trailing60d sigma ==")
shock = O.abs() > 2 * vol60[oil]
days = shock[shock].index
up = days[O.loc[days] > 0]
dn = days[O.loc[days] < 0]
print(f"n shock days = {len(days)} ({len(days)/len(O.dropna())*100:.1f}%), up={len(up)}, down={len(dn)}")

fr = {}
for col in ["XLE", "SPY", "USO", "CL=F"]:
    for h in (1, 2, 5, 10):
        fr[(col, h)] = px[col].shift(-h) / px[col] - 1
for label, dd, sign in [("oil UP shock", up, +1), ("oil DOWN shock", dn, -1)]:
    print(f"\n-- after {label} (n={len(dd)}) --")
    for h in (1, 2, 5, 10):
        row = []
        for col in ("XLE", "SPY", "USO"):
            m = fr[(col, h)].loc[dd].mean() * 100 * sign
            row.append(f"{col} {m:+.3f}%")
        print(f"  h={h:2d}d sign-aligned fwd ret: " + " | ".join(row))

print("\n== XLE sign-aligned forward after oil shock (t-stats) ==")
for h in (1, 5):
    s_ = fr[("XLE", h)].loc[days] * np.sign(O.loc[days])
    s_ = s_.replace([np.inf, -np.inf], np.nan).dropna()
    t = s_.mean() / (s_.std() / np.sqrt(len(s_)))
    print(f"  XLE fwd {h}d: mean {s_.mean()*100:+.3f}%, t={t:+.2f}, n={len(s_)}")
    # up-shock vs down-shock separately
    for lab, dd in [("up", up), ("dn", dn)]:
        s2 = (fr[("XLE", h)].loc[dd] * np.sign(O.loc[dd])).replace([np.inf, -np.inf], np.nan).dropna()
        t2 = s2.mean() / (s2.std() / np.sqrt(len(s2))) if len(s2) > 1 else np.nan
        print(f"    oil {lab}-shock only: mean {s2.mean()*100:+.3f}%, t={t2:+.2f}, n={len(s2)}")

print("\n== USO vol-spike days: |USO ret| > 2.5*trailing sigma -> SPY fwd ==")
shockU = (U.abs() > 2.5 * vol60["USO"])
dU = shockU[shockU].index
f5, f1 = fr[("SPY", 5)], fr[("SPY", 1)]
print(f"n={len(dU)}: next-1d SPY {f1.loc[dU].mean()*100:+.3f}% | "
      f"next-5d SPY {f5.loc[dU].mean()*100:+.3f}% (uncond 5d {f5.mean()*100:+.3f}%)")

print("\n== dollar-oil BOTH-UP days (UUP>0 & USO>0) -> next-1d SPY ==")
both = ((D > 0) & (U > 0))
dB = both[both].index
print(f"n={len(dB)}: next-1d SPY {f1.loc[dB].mean()*100:+.3f}% (uncond {f1.mean()*100:+.3f}%)")

print("\n== 2020 oil-crash window ==")
w = px.loc["2020-03-01":"2020-06-30"]
print(f"CL=F min: {w[oil].min():.2f} on {w[oil].idxmin().date()}; "
      f"XLE maxDD: {(w['XLE']/w['XLE'].cummax()-1).min()*100:.1f}%; "
      f"SPY maxDD: {(w['SPY']/w['SPY'].cummax()-1).min()*100:.1f}%")
