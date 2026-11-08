"""Oil price transmission to FX and equities (read-only research).

Downloads daily yfinance data from 2015 to present plus hourly oil bars, then
computes rolling correlations, regime splits, lead-lag cross-correlations, and
oil shock-day event windows. Saves a JSON summary for the report.
No execution, no orders, no keys.
"""
import json, warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
START = "2015-01-01"
END = "2026-09-21"

TICKERS = {
    "CL": "CL=F", "USO": "USO", "DXY": "DX-Y.NYB", "UUP": "UUP",
    "CAD": "CAD=X", "NOK": "NOK=X", "MXN": "MXN=X",
    "SPY": "SPY", "XLE": "XLE", "TNX": "^TNX", "VIX": "^VIX",
}

def dl(tick, start, end, interval="1d"):
    d = yf.download(tick, start=start, end=end, interval=interval,
                    progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d["Close"].rename(tick)

print("downloading daily...", flush=True)
px = pd.DataFrame({k: dl(v, START, END) for k, v in TICKERS.items()})
print("daily rows:", len(px), "range:", px.index.min(), "->", px.index.max())

# FX pairs are quoted as foreign-per-USD; strength of foreign ccy = -return
for c in ["CAD", "NOK", "MXN"]:
    px[c + "s"] = 1.0 / px[c]   # CAD strength etc. (per-USD basis)

ret = px.pct_change()
# sample bookkeeping
n_obs = int(ret[["CL","SPY","XLE","UUP","DXY","CADs","NOKs","MXNs","TNX","VIX"]].dropna().shape[0])
print("aligned daily obs:", n_obs)

OUT = {"sample": {"start": str(px.index.min().date()), "end": str(px.index.max().date()),
                  "n_daily": n_obs}}

# ---------- 2. Rolling correlations ----------
PAIRS = [
    ("CL", "DXY", "oil-vs-dollar-index"), ("CL", "UUP", "oil-vs-UUP"),
    ("CL", "CADs", "oil-vs-CAD-strength"), ("CL", "NOKs", "oil-vs-NOK-strength"),
    ("CL", "MXNs", "oil-vs-MXN-strength"), ("CL", "SPY", "oil-vs-SPY"),
    ("CL", "XLE", "oil-vs-XLE"), ("CL", "TNX", "oil-vs-10y-yield"),
    ("USO", "SPY", "USO-vs-SPY (xcheck)"), ("SPY", "DXY", "SPY-vs-DXY (xcheck)"),
]
rc = {}
for a, b, name in PAIRS:
    both = ret[[a, b]].dropna()
    c60 = both[a].rolling(60).corr(both[b]).dropna()
    c252 = both[a].rolling(252).corr(both[b]).dropna()
    full = both[a].corr(both[b])
    n = len(both)
    t = full * np.sqrt((n - 2) / max(1e-12, 1 - full ** 2))
    rc[name] = {
        "full_sample_r": round(float(full), 4), "n": n,
        "t_full": round(float(t), 2),
        "r60_mean": round(float(c60.mean()), 4), "r60_min": round(float(c60.min()), 4),
        "r60_max": round(float(c60.max()), 4), "r60_std": round(float(c60.std()), 4),
        "r60_pct_positive": round(float((c60 > 0).mean()), 4),
        "r252_mean": round(float(c252.mean()), 4), "r252_min": round(float(c252.min()), 4),
        "r252_max": round(float(c252.max()), 4), "r252_pct_positive": round(float((c252 > 0).mean()), 4),
    }
OUT["rolling_corr"] = rc

# ---------- 3. Regime splits ----------
med_vix = float(px["VIX"].median())
high_vix = px["VIX"] > med_vix
cl_clean = px["CL"].dropna()                       # own calendar: no union-index NaNs
uptrend = (cl_clean > cl_clean.rolling(200).mean())
uptrend = uptrend.reindex(px.index).fillna(False).astype(bool)  # point-in-time
post2020 = px.index >= "2020-03-01"
regimes = {
    "vix_high": high_vix, "vix_low": ~high_vix,
    "oil_uptrend": uptrend, "oil_downtrend": ~uptrend,
    "pre_2020": ~post2020, "post_2020": post2020,
}
OUT["regimes"] = {"vix_median": round(med_vix, 2),
                  "oil_uptrend_share": round(float(uptrend.mean()), 3)}
for rname, mask in regimes.items():
    d = {}
    for a, b, name in PAIRS:
        both = ret[[a, b]].loc[mask].dropna()
        n = len(both)
        r = both[a].corr(both[b]) if n > 60 else np.nan
        d[name] = {"r": round(float(r), 4) if n > 60 else None, "n": n}
    OUT["regimes"][rname] = d
print("regimes done, med_vix=", med_vix)

# ---------- 4. Lead-lag (daily), lags -5..+5 ----------
# convention: lag k>0 => corr(oil[t-k], x[t]): oil leads by k days
def xcorr(oil, x, lags):
    rows = []
    both = pd.DataFrame({"o": oil, "x": x}).dropna()
    o, x_ = both["o"], both["x"]
    n = len(both)
    for k in lags:
        if k >= 0:
            r = o.shift(k).corr(x_)
        else:
            r = o.corr(x_.shift(-k))
        t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
        rows.append((k, r, t))
    return rows, n

lags_d = list(range(-5, 6))
ll_daily = {}
for xname, label in [("SPY", "oil->SPY"), ("UUP", "oil->UUP"), ("XLE", "oil->XLE")]:
    rows, n = xcorr(ret["CL"], ret[xname], lags_d)
    # Bonferroni across 33 tests (3 pairs x 11 lags): t_crit ~ 3.5 for alpha=0.05
    ll_daily[label] = {
        "n": n,
        "lags": [{"k": k, "r": round(float(r), 4), "t": round(float(t), 2)} for k, r, t in rows],
    }
OUT["leadlag_daily"] = ll_daily

# ---------- hourly ----------
print("downloading hourly USO + CL...", flush=True)
hstart = (px.index.max() - pd.Timedelta(days=75)).strftime("%Y-%m-%d")
uso_h = dl("USO", hstart, END, "1h").rename("USO_h")
cl_h = dl("CL=F", hstart, END, "1h").rename("CL_h")
in1h = pd.read_csv("/home/hatch/workspace/trading-bot/results/datasets/intraday_1h.csv",
                   parse_dates=["timestamp"])
in1h = in1h.set_index("timestamp").sort_index()
xle_h = in1h.loc[in1h["symbol"] == "XLE", "close"].rename("XLE")
spy_h = in1h.loc[in1h["symbol"] == "SPY", "close"].rename("SPY")
H = pd.DataFrame({"USO": uso_h, "CL": cl_h}).join(xle_h).join(spy_h)
# normalize to tz-naive UTC timestamps
def _naive(ix):
    ix = pd.DatetimeIndex(ix)
    if ix.tz is not None:
        ix = ix.tz_convert("UTC").tz_localize(None)
    return ix
H.index = _naive(H.index)
xle_h2 = xle_h.copy(); xle_h2.index = _naive(xle_h.index)
spy_h2 = spy_h.copy(); spy_h2.index = _naive(spy_h.index)
H = pd.DataFrame({"USO": uso_h, "CL": cl_h})
H.index = _naive(H.index)
# CL futures hourly bars land on the hour (:00), equity bars at :30 -> ffill CL
# onto the equity grid (<=30 min staleness, CL trades ~24h)
eq_grid = pd.DatetimeIndex(sorted(set(xle_h2.index) | set(spy_h2.index)))
cl_ff = H["CL"].reindex(H.index.union(eq_grid)).ffill().reindex(eq_grid)
uso_ff = H["USO"].reindex(eq_grid)
H = pd.DataFrame({"USO": uso_ff, "CL": cl_ff}).join(xle_h2).join(spy_h2)
H = H.dropna(subset=["USO", "XLE", "SPY"])
# align CL futures to same timestamps (drop overnight-only bars)
H = H.dropna()
rh = H.pct_change()
OUT["sample"]["n_hourly"] = int(rh.dropna().shape[0])
lags_h = list(range(-10, 11))
ll_h = {}
for (o, x_, label) in [("USO", "XLE", "USO->XLE"), ("USO", "SPY", "USO->SPY"),
                       ("CL", "XLE", "CL->XLE")]:
    rows, n = xcorr(rh[o], rh[x_], lags_h)
    ll_h[label] = {"n": n,
        "lags": [{"k": k, "r": round(float(r), 4), "t": round(float(t), 2)} for k, r, t in rows]}
OUT["leadlag_hourly"] = ll_h
print("hourly n:", OUT["sample"]["n_hourly"])

# ---------- 5. Shock days (per-asset own calendar: no union-NaN poisoning) ----------
cl_r = px["CL"].dropna().pct_change().dropna()
sig60 = cl_r.rolling(60).std().shift(1)          # backward-looking only
shock = (cl_r.abs() > 2.5 * sig60).fillna(False)
shock_days = cl_r.index[shock]
OUT["shocks"] = {"n_shocks": int(len(shock_days)),
                 "shock_dates": [str(d.date()) for d in shock_days]}
print("shocks:", len(shock_days))

evt = {}
for t in ["SPY", "UUP", "TNX", "CADs"]:
    s = px[t].dropna()                           # asset's own trading calendar
    sr = s.pct_change()
    drift = sr.rolling(60).mean().shift(1)       # backward-looking demean
    ar = (sr - drift).dropna()
    pos = {d: i for i, d in enumerate(ar.index)}
    ev = {}
    for w in range(-3, 6):
        vals = []
        for d in shock_days:
            if d in pos:
                i = pos[d] + w
                if 0 <= i < len(ar):
                    vals.append(ar.iloc[i])
        vals = pd.Series(vals).dropna()
        n = len(vals)
        m = vals.mean(); se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        ev[str(w)] = {"mean_ar_bps": round(float(m * 1e4), 1),
                      "se_bps": round(float(se * 1e4), 1) if n > 1 else None,
                      "t": round(float(m / se), 2) if n > 1 and se > 0 else None,
                      "n": n}
    # cumulative CAR[-3,+5]
    car, carse2 = 0.0, 0.0
    for w in range(-3, 6):
        car += ev[str(w)]["mean_ar_bps"] / 1e4
        if ev[str(w)]["se_bps"] is not None:
            carse2 += (ev[str(w)]["se_bps"] / 1e4) ** 2
    evt[t] = {"windows": ev,
              "CAR_bps": round(float(car * 1e4), 1),
              "CAR_se_bps": round(float(np.sqrt(carse2) * 1e4), 1)}
OUT["shock_event"] = evt

with open("/home/hatch/workspace/trading-bot/results/oil_transmission_data.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote oil_transmission_data.json")
