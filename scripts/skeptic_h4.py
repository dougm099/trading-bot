#!/usr/bin/env python3
"""SKEPTIC attack on H4: VIX contrarian.

Runs every attack in the plan:
 1. Sample-size/concentration: remove Mar-Apr 2020, recompute; event clustering
    -> effective n.
 2. Threshold fragility: VIX > 25/30/35 x hold 3/5/10d grid.
 3. Round-number robustness (folded into 2).
 4. Tradability: overnight gaps on event days, cost stress 10/25bps, halt days.
 5. Lookahead: verify signal/timing alignment, auto_adjust consistency.
 6. Multiple comparisons: Bonferroni haircut + permutation p-values.
 7. Base-rate: P(random 5d window > +100bp) vs event distribution.
Extra skepticism: outlier sensitivity (drop top-2 trades), per-event table.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

N_PERM = 20000
RNG = np.random.default_rng(7)


def load():
    spy = yf.download("SPY", start="2015-01-01", end="2026-09-01",
                      interval="1d", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start="2015-01-01", end="2026-09-01",
                      interval="1d", auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
        vix.columns = vix.columns.get_level_values(0)
    df = pd.DataFrame({"o": spy["Open"], "h": spy["High"], "l": spy["Low"],
                       "c": spy["Close"], "v": spy["Volume"],
                       "vix": vix["Close"]}).dropna()
    return df


def events_for(df, thr):
    return df.index[(df["vix"] > thr) & (df["vix"].shift(1) <= thr)]


def fwd_ret(df, ev_dates, entry_lag=1, hold=5):
    """Log return: enter at open of entry_lag-th session after event, exit at
    close hold sessions later."""
    pos = df.index.get_indexer(ev_dates)
    entry = pos + entry_lag
    exit_ = entry + hold
    ok = exit_ < len(df)
    ev = ev_dates[ok]
    entry_px = df["o"].iloc[entry[ok]].to_numpy()
    exit_px = df["c"].iloc[exit_[ok]].to_numpy()
    return pd.Series(np.log(exit_px / entry_px), index=ev)


def effective_n(ev_dates, df, hold=5):
    """Cluster events whose [entry, exit] windows overlap -> effective n."""
    pos = df.index.get_indexer(ev_dates)
    wins = sorted([(p + 1, p + 1 + hold) for p in pos])
    clusters = 1
    cur_end = wins[0][1]
    for s, e in wins[1:]:
        if s <= cur_end:          # overlaps current cluster
            cur_end = max(cur_end, e)
        else:
            clusters += 1
            cur_end = e
    return clusters


def overlap_matrix(ev_dates, df, hold=5):
    pos = np.array(df.index.get_indexer(ev_dates))
    n = len(pos)
    ov = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s1, e1 = pos[i] + 1, pos[i] + 1 + hold
            s2, e2 = pos[j] + 1, pos[j] + 1 + hold
            ov[i, j] = ov[j, i] = max(0, min(e1, e2) - max(s1, s2)) / hold
    return ov


def perm_pvalue(df, ev_dates, hold=5, n_perm=N_PERM):
    """One-sided permutation p: mean of k random non-overlapping h-day windows
    vs observed mean (k = len(events)). Uses all sessions with full window."""
    obs = fwd_ret(df, ev_dates, hold=hold).mean()
    k = len(ev_dates)
    all_pos = np.arange(len(df) - hold - 1)
    r = np.log(df["c"].shift(-hold).to_numpy() / df["o"].shift(-1).to_numpy())
    r = r[1:len(df) - hold]
    all_pos = all_pos[1:]
    cnt = 0
    for _ in range(n_perm):
        samp = RNG.choice(all_pos, size=k, replace=False)
        # enforce non-overlap: greedy keep windows not overlapping previous
        samp = np.sort(samp)
        keep = [samp[0]]
        for s in samp[1:]:
            if s - keep[-1] >= hold:
                keep.append(s)
        if len(keep) < max(5, k // 2):
            continue
        if r[keep].mean() >= obs:
            cnt += 1
    return obs, cnt / n_perm


def main():
    df = load()
    print(f"data: {df.index[0].date()} -> {df.index[-1].date()}, n={len(df)}")

    # ---- reproduce H4 ----
    ev30 = events_for(df, 30)
    r5 = fwd_ret(df, ev30, hold=5) - 2 * 0.0005
    print(f"\n[repro] VIX>30 fresh spikes: n={len(ev30)}, mean5d={r5.mean()*1e4:+.0f}bp "
          f"hit={(r5 > 0).mean():.0%}, cum={np.exp(r5.sum())-1:+.2%}")
    print("  event dates:", ", ".join(d.strftime("%Y-%m-%d") for d in ev30))

    # ---- 1. concentration: drop Mar-Apr 2020 ----
    mask_crash = ~((ev30 >= "2020-03-01") & (ev30 <= "2020-04-30"))
    r5_noc = r5.loc[ev30[mask_crash]]
    n_crash = (~mask_crash).sum()
    print(f"\n[1] Mar-Apr 2020 events: {n_crash}/{len(ev30)}")
    print(f"    ex-crash: n={len(r5_noc)}, mean5d={r5_noc.mean()*1e4:+.0f}bp "
          f"hit={(r5_noc > 0).mean():.0%}, cum={np.exp(r5_noc.sum())-1:+.2%}")

    # clustering / effective n
    cl = effective_n(ev30, df, 5)
    ov = overlap_matrix(ev30, df, 5)
    n_pairs = int((ov > 0).sum() / 2)
    print(f"    overlapping 5d-window pairs: {n_pairs}; effective n (non-overlapping "
          f"clusters): {cl} vs nominal n={len(ev30)}")
    # window spacing
    gaps = np.diff(df.index.get_indexer(ev30))
    print(f"    median gap between events: {np.median(gaps):.0f} sessions; "
          f"events <5 sessions apart: {(gaps < 5).sum()}")

    # ---- 2. threshold x hold grid ----
    print("\n[2] threshold x hold grid (mean net bp/trade, 5bps/side; n in parens)")
    for thr in (25, 30, 35):
        row = []
        for hold in (3, 5, 10):
            ev = events_for(df, thr)
            r = fwd_ret(df, ev, hold=hold) - 2 * 0.0005
            row.append(f"h{hold}: {r.mean()*1e4:+.0f}bp (n={len(r)})")
        print(f"    VIX>{thr}: " + " | ".join(row))

    # ---- outlier sensitivity: drop top-2 trades ----
    top2 = r5.nlargest(2)
    rest = r5.drop(top2.index)
    print(f"\n[outliers] top-2 trades: {[f'{d.date()} {v*1e4:+.0f}bp' for d, v in top2.items()]}")
    print(f"    ex-top2: n={len(rest)}, mean5d={rest.mean()*1e4:+.0f}bp, cum={np.exp(rest.sum())-1:+.2%}")

    # ---- 4. tradability ----
    print("\n[4] tradability")
    pos = df.index.get_indexer(ev30)
    entry_pos = pos + 1
    entry_pos = entry_pos[entry_pos < len(df)]
    overnight = np.log(df["o"].iloc[entry_pos].to_numpy() /
                       df["c"].iloc[entry_pos - 1].to_numpy())
    all_overnight = np.log(df["o"].shift(-0).to_numpy()[1:] / df["c"].to_numpy()[:-1])
    print(f"    |overnight gap| at entry: event-day mean={np.abs(overnight).mean()*1e4:.0f}bp "
          f"max={np.abs(overnight).max()*1e4:.0f}bp vs all-days mean={np.abs(all_overnight).mean()*1e4:.0f}bp")
    # intraday range on event day as slippage proxy
    rng_ev = (df["h"] - df["l"]).iloc[pos] / df["c"].iloc[pos]
    rng_all = ((df["h"] - df["l"]) / df["c"]).mean()
    print(f"    event-day range/close: mean={rng_ev.mean()*100:.1f}% vs all-days {rng_all*100:.1f}%")
    # market-wide halt days in sample (circuit breakers Mar 2020)
    halt_days = pd.to_datetime(["2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18"])
    n_halt = ev30.isin(halt_days).sum()
    print(f"    event dates coinciding with market-wide circuit-breaker halts: {n_halt}")
    # volume spike on entry days
    vol_ratio = (df["v"].iloc[entry_pos] / df["v"].rolling(20).mean().iloc[entry_pos]).mean()
    print(f"    entry-day volume vs 20d avg: {vol_ratio:.1f}x")
    for c in (0.0005, 0.0010, 0.0025):
        rn = r5 + 2 * 0.0005 - 2 * c  # re-add then subtract new cost
        print(f"    cost {c*1e4:.0f}bps/side: mean/trade={rn.mean()*1e4:+.0f}bp, cum={np.exp(rn.sum())-1:+.2%}")

    # ---- 5. lookahead checks ----
    print("\n[5] lookahead")
    # VIX close timing: confirm signal day's SPY close precedes nothing used
    # event defined from vix[t], vix[t-1]; trade at open[t+1]; verify no NaN shift misuse
    manual = (df["vix"] > 30) & (df["vix"].shift(1) <= 30)
    print(f"    event mask reproducible from close-only fields: {bool((manual == df.index.isin(ev30)).all())}")
    # auto_adjust consistency: ratio o/c unaffected by splits/dividends by construction
    print(f"    any zero/negative opens or closes in sample: {bool(((df['o'] <= 0) | (df['c'] <= 0)).any())}")
    # SPY/VIX calendar alignment: both US equities calendar; check date overlap
    print(f"    signal day t uses VIX close (16:00 ET), entry at SPY open t+1 (09:30 ET): no same-day leak by construction")
    # check first VIX>30 fresh events are not on SPY half-days w/ weird opens: list entry dates
    print("    entry opens/traded normally on all events (vol>0):",
          bool((df['v'].iloc[entry_pos] > 0).all()))

    # ---- 6. multiple comparisons ----
    print("\n[6] multiple comparisons")
    obs_mean, p_perm = perm_pvalue(df, ev30, hold=5)
    print(f"    permutation p (mean of 30 random 5d windows >= observed {obs_mean*1e4:.0f}bp): p={p_perm:.4f}")
    # naive t-stat vs unconditional, with effective-n adjustment
    r_all = fwd_ret(df, df.index, hold=5).dropna()
    excess = r5.mean() - r_all.mean()
    se_nom = r5.std() / np.sqrt(len(r5))
    se_eff = r5.std() / np.sqrt(cl)
    from math import erf, sqrt
    def pnorm(z): return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    t_nom, t_eff = excess / se_nom, excess / se_eff
    print(f"    excess mean vs unconditional: {excess*1e4:+.0f}bp; t(nom n=30)={t_nom:+.2f} p={pnorm(t_nom):.4f}; "
          f"t(eff n={cl})={t_eff:+.2f} p={pnorm(t_eff):.4f}")
    print(f"    Bonferroni over 4 round-1 hypotheses: alpha=0.0125. "
          f"{'SURVIVES' if p_perm < 0.0125 else 'DIES'} on permutation p; "
          f"{'SURVIVES' if pnorm(t_eff) < 0.0125 else 'DIES'} on eff-n t-test.")
    # also count the implicit grid: 3 thresholds x 3 holds = 9 variants
    print(f"    (implicit look-elsewhere: thresholdxhold grid = 9 variants -> alpha=0.0056; "
          f"{'SURVIVES' if p_perm < 0.0056 else 'DIES'})")

    # ---- 7. base rate ----
    print("\n[7] base rate")
    r_all = fwd_ret(df, df.index, hold=5).dropna()
    p100 = (r_all > 0.01).mean()
    p100_ev = (r5 > 0.01).mean()
    print(f"    P(random 5d window > +100bp) = {p100:.1%} (n={len(r_all)} windows)")
    print(f"    P(event 5d window > +100bp)  = {p100_ev:.1%} (n={len(r5)})")
    # binomial test of event hit-vs-base
    from math import comb
    k = int((r5 > 0.01).sum())
    n = len(r5)
    pval = sum(comb(n, i) * p100**i * (1 - p100)**(n - i) for i in range(k, n + 1))
    print(f"    binomial P(>={k}/{n} | base {p100:.1%}) = {pval:.4f}")
    # regime-conditional base rate: was 2020-2022 just a good time to be long 5d?
    for name, s, e in [("2015-2019", "2015-01-01", "2019-12-31"),
                       ("2020-2022", "2020-01-01", "2022-12-31"),
                       ("2023-2026", "2023-01-01", "2026-09-01")]:
        rb = r_all.loc[s:e]
        evr = r5.loc[(r5.index >= s) & (r5.index <= e)]
        print(f"    {name}: base P(>+100bp)={ (rb>0.01).mean():.1%}, base mean={rb.mean()*1e4:+.0f}bp | "
              f"events n={len(evr)}, mean={evr.mean()*1e4:+.0f}bp")

    # per-event table
    print("\n[events] date | vix | 5d net bp")
    for d, v in r5.items():
        print(f"    {d.date()} | {df.loc[d,'vix']:5.1f} | {v*1e4:+6.0f}bp")


if __name__ == "__main__":
    main()
    print("\nSKEPTIC DONE")
