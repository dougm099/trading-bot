#!/usr/bin/env python3
"""Round 1: H3 sector 12-1 time-series momentum, H4 VIX contrarian.

H3: 11 sector SPDRs, monthly. Signal = sign of trailing 12m return skipping the
    most recent month (Moskowitz-Ooi-Pedersen 2012). Long-only top-half by
    formation return, hold 1 month, equal weight, 5bps/side. Also long/short
    variant. Regimes: 2015-2019, 2020-2022, 2023-2026. Baseline: equal-weight
    buy-and-hold of the 11 sectors.

H4: daily SPY. When VIX close > 30 (and was <= 30 the prior day, a fresh spike),
    go long SPY at next open, hold 5 trading days. 5bps/side. Report event
    count, mean forward 5d return vs unconditional, per-regime.

No lookahead: formation windows end strictly before holding periods; VIX signal
uses close, trades next open.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
COST_EQ = 0.0005
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLU", "XLK"]
REGIMES = [("2015-2019", "2015-01-01", "2019-12-31"),
           ("2020-2022", "2020-01-01", "2022-12-31"),
           ("2023-2026", "2023-01-01", "2026-09-01")]


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def h3_sector_tsmom():
    print("\n===== H3 sector 12-1 time-series momentum =====")
    px = yf.download(SECTORS, start="2014-01-01", end="2026-09-01",
                     interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
    # monthly closes
    m = px.resample("M").last()
    mret = np.log(m / m.shift(1))
    # formation: 12m return skipping most recent month
    form = mret.rolling(12).sum().shift(1)
    # long-only: top half by formation; long/short: top3 - bottom3
    rets_lo, rets_ls = [], []
    idx = []
    for t in m.index[13:]:
        f = form.loc[t].dropna()
        if len(f) < 8:
            continue
        longs = f.nlargest(6).index
        shorts = f.nsmallest(3).index
        longs3 = f.nlargest(3).index
        # next-month returns
        t_next = m.index[m.index.get_loc(t) + 1] if m.index.get_loc(t) + 1 < len(m) else None
        if t_next is None:
            continue
        r1m = np.log(m.loc[t_next] / m.loc[t])
        rets_lo.append(r1m[longs].mean() - COST_EQ * 2 * 0.5)  # ~half portfolio turns over
        rets_ls.append((r1m[longs3].mean() - r1m[shorts].mean()) / 2 - COST_EQ * 2)
        idx.append(t_next)
    rets_lo = pd.Series(rets_lo, index=idx)
    rets_ls = pd.Series(rets_ls, index=idx)
    bh = mret.mean(axis=1).loc[idx]  # equal-weight sector buy-and-hold
    for name, s, e in REGIMES:
        for label, r in [("TSMOM_longonly", rets_lo), ("TSMOM_longshort", rets_ls),
                         ("EQW_buyhold", bh)]:
            sub = r.loc[s:e]
            if len(sub) < 12:
                print(f"  {name} {label}: n={len(sub)} too small"); continue
            eq = np.exp(sub.cumsum())
            print(f"  {name} {label}: n={len(sub)} tot={eq.iloc[-1]-1:+.2%} "
                  f"Sharpe={ann_sharpe(sub,12):+.2f} maxDD={max_dd(eq):.2%}")
    # full-sample
    eq = np.exp(rets_lo.cumsum())
    print(f"  FULL TSMOM_longonly: tot={eq.iloc[-1]-1:+.2%} Sharpe={ann_sharpe(rets_lo,12):+.2f} "
          f"maxDD={max_dd(eq):.2%}")


def h4_vix_contrarian():
    print("\n===== H4 VIX>30 fresh-spike contrarian (long SPY 5d) =====")
    spy = yf.download("SPY", start="2015-01-01", end="2026-09-01",
                      interval="1d", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start="2015-01-01", end="2026-09-01",
                      interval="1d", auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
        vix.columns = vix.columns.get_level_values(0)
    df = pd.DataFrame({"spy_o": spy["Open"], "spy_c": spy["Close"],
                       "vix": vix["Close"]}).dropna()
    df["fresh_spike"] = (df["vix"] > 30) & (df["vix"].shift(1) <= 30)
    # enter next open, hold 5 sessions
    fwd5 = np.log(df["spy_c"].shift(-5) / df["spy_o"].shift(-1))
    events = df.loc[df["fresh_spike"]].index
    print(f"  fresh VIX>30 spikes: {len(events)}")
    for name, s, e in REGIMES:
        ev = events[(events >= s) & (events <= e)]
        r = fwd5.loc[ev].dropna() - COST_EQ * 2
        base = fwd5.loc[s:e].dropna()  # unconditional 5d forward return
        if len(r) < 3:
            print(f"  {name}: n={len(r)} too small"); continue
        hit = (r > 0).mean()
        print(f"  {name}: n={len(r)} mean5d={r.mean()*1e4:+.0f}bp hit={hit:.0%} "
              f"vs unconditional mean5d={base.mean()*1e4:+.0f}bp")
    r_all = fwd5.loc[events].dropna() - COST_EQ * 2
    eq = np.exp(r_all.cumsum())
    print(f"  FULL: n={len(r_all)} tot5d-trades={eq.iloc[-1]-1:+.2%} hit={(r_all>0).mean():.0%}")


if __name__ == "__main__":
    h3_sector_tsmom()
    h4_vix_contrarian()
    print("\nDONE")
