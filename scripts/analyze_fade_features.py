#!/usr/bin/env python3
"""Feature-pattern analysis for the oil spike-fade trade.

Builds every feature available per trading day and asks which ones
actually discriminate trade outcomes. Descriptive stats only -- n=26
triggered trades is far too small for ML; anything fancier would be theater.

Features (all defined on regular-session 5m bars unless noted):
  gap_pct       open vs prior close (known at open)
  first30_pct   first 30m move (known by 10:00 ET)
  spike_pct     open -> session high (descriptive; high not known in advance)
  trig_time     ET time of first 5m close >= +1.5% vs open
  high_time     ET time of session high
  range_pct     (high-low)/open
  spy_gap/day   SPY overnight gap and session move (market context)
  vix, vix_chg  VIX level and day change
  tension_max, mentions, esc_6h  (iran_watch; only from 2026-09-21)
  dow           day of week
Targets:
  d_pnl_bps     Model D paper P&L (26 triggered days)
  fade_pct      high -> close (all 60 days)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DS = BASE / "results" / "datasets"
COST = 4.0


def bars(sym, period="60d"):
    d = yf.download(sym, period=period, interval="5m", prepost=False, progress=False)
    out = pd.DataFrame({"o": d["Open"].iloc[:, 0], "h": d["High"].iloc[:, 0],
                        "l": d["Low"].iloc[:, 0], "c": d["Close"].iloc[:, 0]})
    out["date"] = out.index.date
    return out


def main():
    uso = bars("USO")
    spy = bars("SPY")
    vix = yf.download("^VIX", period="90d", interval="1d", progress=False)["Close"].iloc[:, 0]

    # prior closes for gaps (daily)
    ud = yf.download("USO", period="90d", interval="1d", progress=False)["Close"].iloc[:, 0]
    sd = yf.download("SPY", period="90d", interval="1d", progress=False)["Close"].iloc[:, 0]
    # map each date -> previous trading day's close (for overnight gaps)
    udates = list(ud.index)
    uprev = {udates[i + 1].date(): float(ud.iloc[i]) for i in range(len(udates) - 1)}
    sdates = list(sd.index)
    sprev = {sdates[i + 1].date(): float(sd.iloc[i]) for i in range(len(sdates) - 1)}
    vixmap = {d.date(): float(vix.loc[d]) for d in vix.index}
    vdates = sorted(vixmap)
    vprev = {vdates[i + 1]: vixmap[vdates[i]] for i in range(len(vdates) - 1)}

    try:
        iw = pd.read_csv(DS / "iran_watch.csv")
        iw["d"] = pd.to_datetime(iw["ts"]).dt.date
        tens = iw.groupby("d").agg(tension_max=("tension", "max"),
                                   mentions=("iran_mentions_24h", "max"),
                                   esc=("escalation_6h", "max"))
    except Exception:
        tens = pd.DataFrame()

    rows = []
    for dt, g in uso.groupby("date"):
        if len(g) < 60 or dt not in uprev:
            continue
        op, hi, lo, cl = (float(g["o"].iloc[0]), float(g["h"].max()),
                          float(g["l"].min()), float(g["c"].iloc[-1]))
        first30 = float(g["c"].iloc[min(5, len(g) - 1)])
        sig = g[g["c"] >= op * 1.015]
        trig_t = sig.index[0].strftime("%H:%M") if len(sig) else None
        sg = spy[spy["date"] == dt]
        sgo = float(sg["o"].iloc[0]) if len(sg) else np.nan
        sgc = float(sg["c"].iloc[-1]) if len(sg) else np.nan
        row = {
            "date": str(dt), "dow": pd.to_datetime(dt).day_name(),
            "gap_pct": round((op / uprev[dt] - 1) * 100, 2),
            "first30_pct": round((first30 / op - 1) * 100, 2),
            "spike_pct": round((hi / op - 1) * 100, 2),
            "fade_pct": round((cl / hi - 1) * 100, 2),
            "day_pct": round((cl / op - 1) * 100, 2),
            "range_pct": round((hi - lo) / op * 100, 2),
            "trig_time": trig_t,
            "high_time": g["h"].idxmax().strftime("%H:%M"),
            "spy_gap": round((sgo / sprev[dt] - 1) * 100, 2) if dt in sprev else np.nan,
            "spy_day": round((sgc / sgo - 1) * 100, 2) if len(sg) else np.nan,
            "vix": round(vixmap.get(dt, np.nan), 2),
            "vix_chg": round(vixmap.get(dt, np.nan) - vprev.get(dt, np.nan), 2),
        }
        if dt in tens.index:
            row["tension_max"] = int(tens.loc[dt, "tension_max"])
            row["mentions"] = int(tens.loc[dt, "mentions"])
            row["esc"] = int(tens.loc[dt, "esc"])
        # Model D paper trade
        if trig_t:
            entry = float(g.loc[sig.index[0], "c"])
            rest = g.loc[sig.index[0]:].iloc[1:]
            stop = entry * 1.0075
            hit = rest[rest["h"] >= stop]
            exit_px, why = (stop, "stop") if len(hit) else (float(g["c"].iloc[-1]), "close")
            row["d_pnl"] = round((entry - exit_px) / entry * 10000 - COST, 1)
            row["d_exit"] = why
        else:
            row["d_pnl"], row["d_exit"] = np.nan, "no-trigger"
        rows.append(row)

    f = pd.DataFrame(rows)
    f.to_csv(DS / "oil_fade_features.csv", index=False)
    print(f"days: {len(f)}, triggered: {f['d_pnl'].notna().sum()}")

    t = f[f["d_pnl"].notna()].copy()
    feats = ["gap_pct", "first30_pct", "spike_pct", "range_pct",
             "spy_gap", "spy_day", "vix", "vix_chg"]
    print("\n=== corr(feature, D-trade P&L)  [n=%d] ===" % len(t))
    for c_ in feats:
        print(f"  {c_:12s} {t[c_].corr(t['d_pnl']):+.3f}")
    print("\n=== corr(feature, fade_pct)  [n=%d, all days] ===" % len(f))
    for c_ in feats:
        print(f"  {c_:12s} {f[c_].corr(f['fade_pct']):+.3f}")

    # bucket: trigger time
    t["trig_bucket"] = pd.to_datetime(t["trig_time"]).dt.strftime("%H:%M").map(
        lambda x: "early<10:00" if x < "10:00" else ("mid10-11:30" if x < "11:30" else "late>11:30"))
    print("\n=== D P&L by trigger-time bucket ===")
    print(t.groupby("trig_bucket")["d_pnl"].agg(["count", "mean", "sum"]).round(1).to_string())
    # bucket: gap sign
    t["gap_bucket"] = t["gap_pct"].map(lambda x: "gap-up>0.5" if x > 0.5 else ("gap-dn<-0.5" if x < -0.5 else "flat"))
    print("\n=== D P&L by overnight gap bucket ===")
    print(t.groupby("gap_bucket")["d_pnl"].agg(["count", "mean", "sum"]).round(1).to_string())
    # bucket: spike quartile
    t["spike_q"] = pd.qcut(t["spike_pct"], 2, labels=["smaller", "bigger"])
    print("\n=== D P&L by spike size (median split) ===")
    print(t.groupby("spike_q", observed=True)["d_pnl"].agg(["count", "mean", "sum"]).round(1).to_string())
    # dow
    print("\n=== D P&L by day of week ===")
    print(t.groupby("dow")["d_pnl"].agg(["count", "mean", "sum"]).round(1).to_string())
    # winners vs losers: what did winners look like at entry?
    w = t[t["d_pnl"] > 0]; l = t[t["d_pnl"] <= 0]
    print(f"\n=== winners (n={len(w)}) vs losers (n={len(l)}) at entry ===")
    for c_ in ["gap_pct", "first30_pct", "spike_pct", "spy_day", "vix"]:
        print(f"  {c_:12s} win {w[c_].mean():+7.2f} | lose {l[c_].mean():+7.2f}")
    # iran features (tiny n)
    ti = t[t["tension_max"].notna()]
    if len(ti):
        print(f"\n=== iran-feature days in triggered set (n={len(ti)}) ===")
        print(ti[["date", "tension_max", "mentions", "esc", "d_pnl"]].to_string(index=False))


if __name__ == "__main__":
    main()
