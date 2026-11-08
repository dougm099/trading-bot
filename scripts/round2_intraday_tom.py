#!/usr/bin/env python3
"""Round 2: H6 intraday momentum (Gao/Han/Li/Zhou JFE 2018 adaptation),
H7 turn-of-month.

H6: paper: last-30min return continues first-30min return (intraday momentum),
stronger when VIX>20. Adaptation to available data: yfinance 1h bars (730d);
signal = first trading hour (09:30-10:30 ET) log-return sign; trade last hour
(15:00-16:00 ET) in the same direction. Variants: with/without VIX>20 filter,
and reversal variant. 5bps/side, 1 round trip/day max. Regimes: 2024, 2025-2026
(hourly data starts Oct 2023; daily VIX from 2023).

H7: turn-of-month: long SPY from close of last trading day of month to close
of 3rd trading day of new month (Lakonishok-Smidt / TOM literature). Daily
2015-2026, 5bps/side. Baseline: buy-and-hold SPY. Per-regime.

No lookahead: H6 signal uses first-hour bar only, trades last-hour bar;
H7 uses month-end calendar known in advance.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
COST = 0.0005


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def h6_intraday_momentum():
    print("\n===== H6 intraday momentum: first-hour -> last-hour (SPY) =====")
    h = yf.download("SPY", period="730d", interval="1h", auto_adjust=True,
                    progress=False)
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    h.index = pd.to_datetime(h.index, utc=True).tz_convert("America/New_York")
    h["ret"] = np.log(h["Close"] / h["Close"].shift(1))
    h["date"] = h.index.date
    h["hour"] = h.index.hour
    vix = yf.download("^VIX", period="730d", interval="1d", auto_adjust=True,
                      progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix.index = pd.to_datetime(vix.index).date
    vclose = vix["Close"]
    recs = []
    for d, g in h.groupby("date"):
        g = g.sort_index()
        first = g[(g["hour"] == 9) | ((g["hour"] == 10) & (g.index.minute < 30))]
        last = g[g["hour"] == 15]
        if len(first) == 0 or len(last) == 0:
            continue
        r_first = float(np.log(first["Close"].iloc[-1] / first["Open"].iloc[0]))
        # last hour: 15:00 -> 16:00 close
        r_last = float(np.log(last["Close"].iloc[-1] / last["Open"].iloc[0]))
        vx = vclose.get(d, np.nan)
        recs.append((d, r_first, r_last, vx))
    df = pd.DataFrame(recs, columns=["date", "r_first", "r_last", "vix"]).dropna()
    df["date"] = pd.to_datetime(df["date"])
    print(f"  n={len(df)} trading days, {df['date'].min().date()} -> {df['date'].max().date()}")
    corr = df["r_first"].corr(df["r_last"])
    print(f"  corr(first-hour, last-hour) = {corr:+.3f}")
    for label, sub, filt in [
        ("mom_all", df, None),
        ("mom_vix20", df, df["vix"] > 20),
        ("rev_all", df, None),
    ]:
        s = sub if filt is None else df[filt]
        if len(s) < 30:
            print(f"  {label}: n={len(s)} too small"); continue
        sig = np.sign(s["r_first"].values)
        r = sig * s["r_last"].values - COST * 2 if "rev" not in label else -sig * s["r_last"].values - COST * 2
        eq = np.exp(np.cumsum(r))
        mid = len(s) // 2
        r1, r2 = r[:mid], r[mid:]
        print(f"  {label}: n={len(s)} tot={eq[-1]-1:+.2%} Sharpe={ann_sharpe(r,252):+.2f} "
              f"maxDD={max_dd(eq):.2%} | H1 Sh={ann_sharpe(r1,252):+.2f} H2 Sh={ann_sharpe(r2,252):+.2f}")
    # baseline: always long last hour
    r_bh = df["r_last"].values - COST * 2
    eq = np.exp(np.cumsum(r_bh))
    print(f"  baseline long-last-hour: tot={eq[-1]-1:+.2%} Sharpe={ann_sharpe(r_bh,252):+.2f}")


def h7_turn_of_month():
    print("\n===== H7 turn-of-month (long SPY month-end -> 3rd day) =====")
    d = yf.download("SPY", start="2015-01-01", end="2026-09-01", interval="1d",
                    auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d["ret"] = np.log(d["Close"] / d["Close"].shift(1))
    d["ym"] = d.index.to_period("M")
    d["is_month_end"] = d["ym"] != d["ym"].shift(-1)
    # position: enter at close of last trading day, hold through close of 3rd trading day of new month
    pos = pd.Series(0, index=d.index)
    month_ends = d.index[d["is_month_end"]]
    rets = []
    for me in month_ends:
        i = d.index.get_loc(me)
        j = min(i + 3, len(d) - 1)
        window = d["ret"].iloc[i + 1:j + 1]
        rets.append(window.sum() - COST * 2)
    rets = pd.Series(rets, index=month_ends)
    bh = d["ret"]
    for name, s, e in [("2015-2019", "2015-01-01", "2019-12-31"),
                       ("2020-2022", "2020-01-01", "2022-12-31"),
                       ("2023-2026", "2023-01-01", "2026-09-01")]:
        sub = rets.loc[s:e]
        bh_sub = bh.loc[s:e]
        if len(sub) < 12:
            print(f"  {name}: n={len(sub)} too small"); continue
        eq, eqb = np.exp(sub.cumsum()), np.exp(bh_sub.cumsum())
        print(f"  {name}: n={len(sub)} TOM tot={eq.iloc[-1]-1:+.2%} Sharpe={ann_sharpe(sub,12):+.2f} "
              f"maxDD={max_dd(eq):.2%} | buy-hold tot={eqb.iloc[-1]-1:+.2%} Sharpe={ann_sharpe(bh_sub,252):+.2f}")
    eq = np.exp(rets.cumsum())
    print(f"  FULL TOM: n={len(rets)} tot={eq.iloc[-1]-1:+.2%} Sharpe={ann_sharpe(rets,12):+.2f} "
          f"maxDD={max_dd(eq):.2%} hit={(rets>0).mean():.0%}")


if __name__ == "__main__":
    h6_intraday_momentum()
    h7_turn_of_month()
    print("\nDONE")
