#!/usr/bin/env python3
"""VIX fear-spike extension study (2011+).

Extends the Thread 4 survivor: VIX curve inversion (VIX3M/VIX < thr)
-> long equity index, hold N trading days.

Design (pre-registered primary config: SPY / 20d / thr=1.0 — the
Thread 4 winner; everything else is exploratory and reported as such):
- Data: ^VIX, ^VIX3M, SPY, QQQ, IWM daily, 2011-01-01 -> present (yfinance).
- Signal: slope = VIX3M/VIX at day-t close < threshold.
- Execution: enter at NEXT OPEN after signal day; exit at close N trading
  days after entry (no lookahead; 10 bps round-trip cost).
- Episodes: signals firing while already in position are merged into the
  current episode (one episode = one independent observation).
- Stats per config: per-trade mean/hit/worst/best; mean EXCESS vs
  buy-and-hold over the identical hold window (self-benchmarked);
  regime splits 2011-15 / 2016-19 / 2020-22 / 2023-26; first/second half.
- Overlay test (H26 retest): buy-hold equity + 100% extra long during
  episodes vs pure buy-hold -> compare Sharpe AND total return.
- Convergence check: Prometheus H4 variant (fresh VIX>30 -> 5d hold)
  on the long sample.

Research only. No orders, no keys.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path(__file__).resolve().parent.parent / "results"
COST_RT = 0.0010  # 10 bps round-trip
REGIMES = [("2011-15", "2011-01-01", "2015-12-31"),
           ("2016-19", "2016-01-01", "2019-12-31"),
           ("2020-22", "2020-01-01", "2022-12-31"),
           ("2023-26", "2023-01-01", "2030-01-01")]
ASSETS = ["SPY", "QQQ", "IWM"]
HOLDS = [10, 20, 30]
THRS = [1.05, 1.0, 0.95]
PRIMARY = ("SPY", 20, 1.0)


def fetch():
    raw = yf.download(["^VIX", "^VIX3M"] + ASSETS, start="2011-01-01",
                      end="2026-09-21", progress=False, auto_adjust=True)
    out = {}
    for t in ["^VIX", "^VIX3M"] + ASSETS:
        df = pd.DataFrame({"open": raw["Open"][t], "close": raw["Close"][t]}).dropna()
        out[t] = df
    return out


def episodes(opens, closes, slope, thr, hold):
    """Return list of (entry_date, exit_date, gross_ret, bh_ret) per episode."""
    sig = slope < thr
    sig = sig.reindex(closes.index).fillna(False)
    idx = closes.index
    pos = np.zeros(len(idx), dtype=bool)
    eps, i = [], 0
    while i < len(idx):
        if sig.iloc[i] and not pos[i]:
            entry = i + 1
            if entry >= len(idx):
                break
            exit_ = min(entry + hold - 1, len(idx) - 1)
            pos[entry:exit_ + 1] = True
            er = closes.iloc[exit_] / opens.iloc[entry] - 1.0
            bh = closes.iloc[exit_] / closes.iloc[entry - 1] - 1.0
            eps.append((idx[entry], idx[exit_], er, bh))
            i = exit_ + 1
        else:
            i += 1
    return eps


def episode_stats(eps):
    if not eps:
        return None
    net = np.array([e[2] for e in eps]) - COST_RT
    bh = np.array([e[3] for e in eps])
    exc = net - bh
    return dict(n=len(eps), mean_net=net.mean(), hit=(net > 0).mean(),
                worst=net.min(), best=net.max(),
                mean_excess=exc.mean(),
                t_excess=exc.mean() / (exc.std(ddof=1) / np.sqrt(len(exc)))
                if exc.std(ddof=1) > 0 else np.nan)


def run_grid(data):
    slope = (data["^VIX3M"]["close"] / data["^VIX"]["close"]).dropna()
    print(f"slope days: {len(slope)} ({slope.index[0].date()} -> {slope.index[-1].date()})")
    print(f"backwardation days @thr=1.0: {(slope < 1.0).sum()} "
          f"({(slope < 1.0).mean()*100:.1f}%)")
    rows = []
    for asset, hold, thr in itertools.product(ASSETS, HOLDS, THRS):
        d = data[asset]
        eps = episodes(d["open"], d["close"], slope, thr, hold)
        s = episode_stats(eps)
        if s is None:
            continue
        rec = dict(asset=asset, hold=hold, thr=thr, **s)
        # per-regime mean excess
        for name, s0, s1 in REGIMES:
            sub = [e for e in eps if s0 <= str(e[0].date()) <= s1]
            ss = episode_stats(sub)
            rec[f"n_{name}"] = ss["n"] if ss else 0
            rec[f"ex_{name}"] = (ss["mean_excess"] if ss else np.nan)
        # first/second half
        mid = len(eps) // 2
        for tag, sub in (("h1", eps[:mid]), ("h2", eps[mid:])):
            ss = episode_stats(sub)
            rec[f"ex_{tag}"] = ss["mean_excess"] if ss else np.nan
        rows.append(rec)
    return pd.DataFrame(rows), slope


def overlay_test(data, slope, asset="SPY", hold=20, thr=1.0):
    """H26 retest: buy-hold vs buy-hold + 100% extra long during episodes."""
    d = data[asset]
    closes, opens = d["close"], d["open"]
    sig = (slope < thr).reindex(closes.index).fillna(False)
    idx = closes.index
    extra = np.zeros(len(idx))
    i = 0
    while i < len(idx):
        if sig.iloc[i]:
            entry = i + 1
            if entry >= len(idx):
                break
            exit_ = min(entry + hold - 1, len(idx) - 1)
            extra[entry:exit_ + 1] = 1.0
            i = exit_ + 1
        else:
            i += 1
    r = closes.pct_change().fillna(0)
    # extra leg: enter at open after signal -> approximate with close-to-close
    # shifted one day (conservative: misses the entry-day move entirely)
    r_extra = extra * r.shift(-0)  # placeholder, replaced below
    # proper: extra-leg daily return = r on days in position, but entry uses
    # open-to-close of entry day and exit skips post-exit. Close-to-close on
    # in-position days is exact except entry/exit edges; adjust below.
    inpos = extra.astype(bool)
    r_leg = r.copy()
    r_leg[~inpos] = 0.0
    # entry edge: replace entry-day close-to-close with open-to-close
    entries = np.where(np.diff(np.r_[0, extra]) == 1)[0]
    for e in entries:
        r_leg.iloc[e] = opens.iloc[e] / closes.iloc[e - 1] - 1.0 if e > 0 else 0.0
    # cost: 10 bps per episode round trip
    n_ep = len(entries)
    r_leg.iloc[entries] -= COST_RT
    bh_eq = (1 + r).cumprod()
    ov_eq = (1 + r + r_leg).cumprod()

    def sharpe(eq):
        rr = eq.pct_change().dropna()
        return rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else np.nan

    def mdd(eq):
        return ((eq / eq.cummax()) - 1).min()

    return dict(n_episodes=n_ep,
                bh_ret=bh_eq.iloc[-1] - 1, bh_sharpe=sharpe(bh_eq), bh_mdd=mdd(bh_eq),
                ov_ret=ov_eq.iloc[-1] - 1, ov_sharpe=sharpe(ov_eq), ov_mdd=mdd(ov_eq))


def h4_variant(data, slope, asset="SPY"):
    """Prometheus H4 on the long sample: fresh VIX>30 -> 5d hold."""
    d = data[asset]
    closes, opens = d["close"], d["open"]
    vix = data["^VIX"]["close"].reindex(closes.index).ffill()
    fresh = (vix > 30) & (vix.shift(1) <= 30)
    idx = closes.index
    eps, i = [], 0
    while i < len(idx):
        if fresh.iloc[i]:
            entry = i + 1
            if entry >= len(idx):
                break
            exit_ = min(entry + 4, len(idx) - 1)
            er = closes.iloc[exit_] / opens.iloc[entry] - 1.0 - COST_RT
            bh = closes.iloc[exit_] / closes.iloc[entry - 1] - 1.0
            eps.append((idx[entry], er, bh))
            i = exit_ + 1
        else:
            i += 1
    s = episode_stats([(e[0], e[0], e[1] + COST_RT, e[2]) for e in eps])
    if s:
        net = np.array([e[1] for e in eps])
        bh = np.array([e[2] for e in eps])
        s["mean_net"] = net.mean()
        s["mean_excess"] = (net - bh).mean()
    return s, len(eps)


def main():
    data = fetch()
    grid, slope = run_grid(data)
    grid.to_csv(OUT / "vix_extension_grid.csv", index=False)

    print("\n=== GRID (mean NET return/trade, hit rate, mean excess vs BH) ===")
    g = grid.copy()
    g["mark"] = np.where((g.asset == PRIMARY[0]) & (g.hold == PRIMARY[1])
                         & (g.thr == PRIMARY[2]), " <-- PRIMARY (pre-registered)", "")
    print(g[["asset", "hold", "thr", "n", "mean_net", "hit", "mean_excess",
             "t_excess", "mark"]].to_string(index=False,
             formatters={"mean_net": "{:+.3%}".format, "hit": "{:.0%}".format,
                         "mean_excess": "{:+.3%}".format, "t_excess": "{:+.2f}".format}))

    print("\n=== PRIMARY CONFIG regime / half splits (mean excess vs BH) ===")
    p = grid[(grid.asset == PRIMARY[0]) & (grid.hold == PRIMARY[1])
             & (grid.thr == PRIMARY[2])].iloc[0]
    for name, _, _ in REGIMES:
        print(f"  {name}: n={int(p[f'n_{name}']):3d}  mean_excess={p[f'ex_{name}']:+.3%}")
    print(f"  first half: {p['ex_h1']:+.3%}   second half: {p['ex_h2']:+.3%}")

    print("\n=== OVERLAY TEST (H26 retest, 2011+, SPY 20d thr=1.0) ===")
    ov = overlay_test(data, slope)
    print(f"  episodes: {ov['n_episodes']}")
    print(f"  buy-hold:        ret {ov['bh_ret']:+.1%}  Sharpe {ov['bh_sharpe']:.2f}  mDD {ov['bh_mdd']:.1%}")
    print(f"  buy-hold+overlay: ret {ov['ov_ret']:+.1%}  Sharpe {ov['ov_sharpe']:.2f}  mDD {ov['ov_mdd']:.1%}")

    print("\n=== H4 VARIANT retest (fresh VIX>30 -> 5d, 2011+, SPY) ===")
    s, n = h4_variant(data, slope)
    if s:
        print(f"  n={n}  mean_net={s['mean_net']:+.3%}  hit={s['hit']:.0%}  "
              f"mean_excess={s['mean_excess']:+.3%}  t={s['t_excess']:+.2f}")

    print("\nMultiple-comparison note: 27 configs tested; primary pre-registered.")
    print("Bonferroni bar for exploratory t-stats: |t| >= ~2.9 (27 tests).")


if __name__ == "__main__":
    main()
