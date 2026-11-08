"""Fit Normal vs Student-t to return series. Descriptive, not predictive."""
import numpy as np, pandas as pd, glob, os
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def logrets(s):
    return np.log(s / s.shift(1)).dropna().values

series = {}
# SPY hourly
px = pd.read_csv(f"{BASE}/results/prices_1h.csv", index_col=0, parse_dates=True)
series["SPY hourly"] = logrets(px["SPY"])
series["SPY daily"] = logrets(px["SPY"].resample("D").last())
# EURUSD daily
fx = pd.read_csv(f"{BASE}/results/raw_fx/EURUSD_daily.csv", index_col=0, parse_dates=True)
col = [c for c in fx.columns if "close" in c.lower()][0]
series["EURUSD daily"] = logrets(fx[col])
# BTC daily (yfinance)
b = glob.glob(f"{BASE}/results/raw_xs_crypto/yf_BTC.csv")
if b:
    df = pd.read_csv(b[0], index_col=0, parse_dates=True)
    c2 = "price" if "price" in df.columns else [c for c in df.columns if "close" in c.lower()][0]
    series["BTC daily"] = logrets(df[c2])

print(f"{'series':<14}{'n':>8}{'skew':>8}{'kurt':>8}{'df(t)':>8}{'AIC N':>12}{'AIC t':>12}{'dAIC':>8}")
for name, r in series.items():
    r = r[np.isfinite(r)]
    mu, sd = stats.norm.fit(r)
    ll_n = np.sum(stats.norm.logpdf(r, mu, sd)); aic_n = 4 - 2 * ll_n
    df, loc, sc = stats.t.fit(r)
    ll_t = np.sum(stats.t.logpdf(r, df, loc, sc)); aic_t = 6 - 2 * ll_t
    print(f"{name:<14}{len(r):>8}{stats.skew(r):>8.2f}{stats.kurtosis(r):>8.1f}"
          f"{df:>8.2f}{aic_n:>12.0f}{aic_t:>12.0f}{aic_n - aic_t:>8.0f}")

print("\n--- Tail check: P(|r| > 4 sigma) per 10k obs ---")
for name, r in series.items():
    r = r[np.isfinite(r)]
    s = r.std()
    emp = np.mean(np.abs(r) > 4 * s) * 10000
    pnorm = 2 * (1 - stats.norm.cdf(4)) * 10000
    df, loc, sc = stats.t.fit(r)
    pt = 2 * (1 - stats.t.cdf(4 * s, df, loc, sc)) * 10000
    print(f"{name:<14} empirical {emp:>7.1f}   normal {pnorm:>6.2f}   t(df={df:.1f}) {pt:>7.1f}")

print("\n--- 99% VaR / ES (one-period loss), empirical vs fitted ---")
for name, r in series.items():
    r = r[np.isfinite(r)]
    mu, sd = stats.norm.fit(r)
    df, loc, sc = stats.t.fit(r)
    var_e, var_n, var_t = -np.quantile(r, .01), -(mu + sd * stats.norm.ppf(.01)), -(loc + sc * stats.t.ppf(.01, df))
    es_e = -r[r <= np.quantile(r, .01)].mean()
    tail = r[r <= mu + sd * stats.norm.ppf(.01)]
    es_n = -(mu - sd * stats.norm.pdf(stats.norm.ppf(.01)) / .01)
    q = stats.t.ppf(.01, df)
    es_t = -(loc - sc * (stats.t.pdf(q, df) / .01) * ((df + q ** 2) / (df - 1)))
    print(f"{name:<14} VaR  emp {var_e*100:6.2f}%  N {var_n*100:6.2f}%  t {var_t*100:6.2f}%   "
          f"ES  emp {es_e*100:6.2f}%  N {es_n*100:6.2f}%  t {es_t*100:6.2f}%")
