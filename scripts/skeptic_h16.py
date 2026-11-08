#!/usr/bin/env python3
"""SKEPTIC attack on H16: short-VXX carry.

Kill attempts:
 1. DATA INTEGRITY: VXX reverse-split 8x in history (1-for-4: 08/09/2016,
    08/23/2017, 04/23/2021, 03/07/2023, 07/24/2024 per splithistory.com/OCC).
    Check auto_adjust handled them (no artificial jumps in adjusted series;
    jumps should appear ONLY in the raw unadjusted series). Also check the
    Jan-2019 redemption/re-issue of VXX (Barclays redeemed the old ETN and
    re-issued Series B) for a discontinuity/gap.
 2. BORROW COST REALITY: rerun with 2/5/10/25%/yr flat drag; discuss why a
    flat drag is dishonest (fees spike exactly when the short hurts most).
 3. FEB 2018 VOLMAGEDDON: day-by-day reconstruction Jan-31->Feb-28 2018.
    Margin math for a Reg-T short: 50% initial, 30% maintenance => broker
    margin call when price moves ~15.4% against the short (worse with
    typical 40% broker maintenance: ~7.1%). Show the intra-month adverse
    excursion vs the backtest's close-to-close fiction.
 4. MAR 2020: identify the dodged month, show unfiltered return that month.
 5. IMPLEMENTABILITY: no broker calls — reasoned from public knowledge.
 6. MULTIPLE COMPARISONS: t vs Bonferroni over ~16 hypotheses; note the
    1.5x filter threshold was chosen after seeing the 2x version (snooping).
 7. OUTLIER DEPENDENCE: drop best 3 months, recompute Sharpe/total.
 8. RECENCY: is 2023-26 one long calm-vol regime? Compare annualized log
    returns per regime; stress what a regime snap does.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

COST_EQ = 0.0005
# known VXX 1-for-4 reverse split dates (splithistory.com / OCC memos)
SPLITS = ["2016-08-09", "2017-08-23", "2021-04-23", "2023-03-07", "2024-07-24"]


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(np.asarray(equity, dtype=float))
    return float(np.min(np.asarray(equity) / peak - 1))


def load_vxx():
    adj = yf.download("VXX", start="2015-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    raw = yf.download("VXX", start="2015-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=False, progress=False)
    for d in (adj, raw):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
    return adj, raw


def h16(prc, borrow_yr, cost2=COST_EQ * 2):
    """Replicate H16 (filtered, no-lookahead). Returns monthly log-ret series."""
    c = prc["Close"]
    ma60 = c.rolling(60).mean()
    m = c.resample("M").last()
    rets, skipped = [], []
    drag = borrow_yr / 12
    for i in range(len(m) - 1):
        t0, t1 = m.index[i], m.index[i + 1]
        c0, ma0 = c.loc[:t0].iloc[-1], ma60.loc[:t0].iloc[-1]
        if c0 > 1.5 * ma0:
            rets.append(0.0)
            skipped.append(t1)
        else:
            rets.append(-float(np.log(m.iloc[i + 1] / m.iloc[i])) - drag - cost2)
    return pd.Series(rets, index=m.index[1:]), skipped, m


def main():
    adj, raw = load_vxx()
    print("===== 1. DATA INTEGRITY: reverse splits =====")
    ar = np.log(adj["Close"] / adj["Close"].shift(1))
    rr = np.log(raw["Close"] / raw["Close"].shift(1))
    print(f"  adjusted daily max log-move: {ar.max():+.3f} on {ar.idxmax().date()} "
          f"(>1.0 would scream unadjusted split)")
    print(f"  raw (unadjusted) daily max log-move: {rr.max():+.3f} on {rr.idxmax().date()}")
    for s in SPLITS:
        d = pd.Timestamp(s)
        idxa = adj.index.get_indexer([d], method="nearest")[0]
        da = adj.index[idxa]
        print(f"  {s} (nearest px day {da.date()}): adjusted close "
              f"{adj['Close'].iloc[idxa]:.2f}, adj 1d log-ret "
              f"{ar.iloc[idxa]:+.3f} | raw 1d log-ret {rr.iloc[idxa]:+.3f}")
    # Jan 2019 redemption/re-issue window
    sub = adj.loc["2019-01-28":"2019-02-05"]
    print("  Jan-2019 redemption window (adjusted closes):")
    for dt, row in sub.iterrows():
        print(f"    {dt.date()} close={row['Close']:.2f}")
    print(f"  biggest 1d adjusted jump in sample (ex-splits): {ar.dropna().nlargest(3)}")

    print("\n===== 2. BORROW-COST STRESS =====")
    for b in (0.02, 0.05, 0.10, 0.25):
        pr, _, _ = h16(adj, b)
        eq = np.exp(pr.cumsum())
        print(f"  borrow {b:.0%}/yr: n={len(pr)} tot={eq.iloc[-1]-1:+.2%} "
              f"Sh={ann_sharpe(pr, 12):+.2f} maxDD={max_dd(eq):.2%}")

    print("\n===== 3. FEB 2018 VOLMAGEDDON, day by day =====")
    p0 = float(adj["Close"].loc[:"2018-01-31"].iloc[-1])
    w = adj.loc["2018-01-31":"2018-02-28", ["Open", "High", "Low", "Close"]]
    print(f"  entry (Jan-31-2018 close, adj): {p0:.2f}")
    worst = 0.0
    for dt, row in w.iterrows():
        adv = row["High"] / p0 - 1
        worst = max(worst, adv)
        flag = "  <-- margin call level for Reg-T short (30% maint: +15.4%)" if adv > 0.154 else ""
        flag = flag or ("  <-- broker 40% maint call (+7.1%)" if adv > 0.071 else "")
        print(f"  {dt.date()} O={row['Open']:8.2f} H={row['High']:8.2f} "
              f"L={row['Low']:8.2f} C={row['Close']:8.2f}  intraday adverse {adv:+.1%}{flag}")
    mcc = float(w.loc[pd.Timestamp("2018-02-28"), "Close"]) if pd.Timestamp("2018-02-28") in w.index else np.nan
    print(f"  worst intraday adverse excursion: {worst:+.1%}")
    print(f"  backtest's close-to-close Feb-2018 short return: {-np.log(mcc/p0):+.2%} "
          f"(hides everything above)")

    print("\n===== 3b. EVERY holding month vs margin (intraday reality check) =====")
    # Reg-T short: initial margin 50%. Equity E(t) = (1.5*P0 - Pt)*Q.
    # 30% NYSE maintenance call when Pt/P0 > 1.5/1.3 = 1.1538 (adverse +15.4%).
    # Typical broker 40% maintenance call when Pt/P0 > 1.5/1.4 = 1.0714 (+7.1%).
    # Equity hits ZERO (total wipeout, forced liquidation) at +50%.
    c = adj["Close"]
    m = c.resample("M").last()
    rows = []
    for i in range(len(m) - 1):
        t0, t1 = m.index[i], m.index[i + 1]
        p0 = float(m.iloc[i])
        days = adj.loc[t0:t1]
        adv = float((days["High"] / p0).max() - 1)
        r = -float(np.log(m.iloc[i + 1] / m.iloc[i]))
        rows.append((t1.strftime("%Y-%m"), adv, r))
    for lab, thr in [("broker 40% call (+7.1%)", 0.071),
                     ("Reg-T 30% call (+15.4%)", 0.154),
                     ("equity ZERO / wipeout (+50%)", 0.50)]:
        hit = [x for x in rows if x[1] > thr]
        logsum = sum(x[2] for x in rows if x[1] <= thr)
        print(f"  {lab}: breached in {len(hit)}/{len(rows)} holding months")
        if len(hit) <= 8:
            for d, a, r in hit:
                print(f"    {d}: worst intraday adverse {a:+.1%}, backtest claims {r:+.2%}")
        print(f"    log P&L earned ONLY in survivable months (<=thr): {logsum:+.3f} "
              f"vs backtest total {sum(x[2] for x in rows):+.3f}")
    # March 2020 detail
    print("\n  March 2020 day by day (entry Feb-29-2020 close):")
    p0 = float(adj["Close"].loc[:"2020-02-29"].iloc[-1])
    w = adj.loc["2020-02-29":"2020-03-31", ["High", "Close"]]
    for dt, row in w.iterrows():
        adv = row["High"] / p0 - 1
        print(f"    {dt.date()} C={row['Close']:8.2f} intraday adverse {adv:+.1%}")
    pr2, skipped, m = h16(adj, 0.02)
    print(f"  skipped months: {[t.strftime('%Y-%m') for t in skipped]}")
    raw_m = -np.log(m / m.shift(1)).dropna() - 0.02 / 12 - COST_EQ * 2
    for t in skipped:
        print(f"  unfiltered {t.strftime('%Y-%m')} return: {raw_m.loc[t]:+.2%} "
              f"(what the filter avoided)")
    unskip = raw_m.loc[[t for t in pr2.index if t not in skipped]]
    print(f"  n={len(pr2)}, total dodged log-return avoided: "
          f"{raw_m.loc[skipped].sum():+.4f}")

    print("\n===== 6. MULTIPLE COMPARISONS =====")
    sh = ann_sharpe(pr2, 12)
    yrs = len(pr2) / 12
    t = sh * np.sqrt(yrs)
    print(f"  t = {sh:.2f}*sqrt({yrs:.2f}y) = {t:.2f}")
    import math
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(t))
    thr = norm.ppf(1 - 0.05 / 16 / 2)
    print(f"  two-sided p = {p:.4f}; Bonferroni 0.05/16: required |z| >= {thr:.2f}; "
          f"observed {t:.2f} -> {'PASS' if t > thr else 'FAIL'}")
    # also: the 1.5x filter was picked after the 2x version was tried -> data snooping
    pr20, _, _ = h16(adj, 0.02)
    print("  (filter threshold 1.5x chosen after 2x version existed: "
          "one more implicit comparison)")

    print("\n===== 7. OUTLIER DEPENDENCE =====")
    eq = np.exp(pr2.cumsum())
    top3 = pr2.nlargest(3)
    print(f"  best 3 months: {[(t.strftime('%Y-%m'), f'{v:+.2%}') for t, v in top3.items()]}")
    print(f"  their log-sum share: {top3.sum()/pr2.sum():.1%} of total log P&L")
    prx = pr2.drop(top3.index)
    eqx = np.exp(prx.cumsum())
    print(f"  drop best 3: tot={eqx.iloc[-1]-1:+.2%} Sh={ann_sharpe(prx, 12):+.2f} "
          f"maxDD={max_dd(eqx):.2%}")
    top1 = pr2.nlargest(1)
    pr1 = pr2.drop(top1.index)
    print(f"  drop best 1: tot={np.exp(pr1.cumsum()).iloc[-1]-1:+.2%} "
          f"Sh={ann_sharpe(pr1, 12):+.2f}")

    print("\n===== 8. RECENCY =====")
    for name, s, e in [("2015-19", "2015", "2019"), ("2020-22", "2020", "2022"),
                       ("2023-26", "2023", "2026")]:
        sub = pr2.loc[s:e]
        yrs_ = len(sub) / 12
        print(f"  {name}: n={len(sub)} tot={np.exp(sub.cumsum()).iloc[-1]-1:+.2%} "
              f"Sh={ann_sharpe(sub, 12):+.2f} ann-log {sub.sum()/yrs_:+.3f}/yr")
    print("  NOTE: strategy is structurally 'sell insurance'; regime means:")
    vix = yf.download("^VIX", start="2015-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vm = vix["Close"].resample("M").mean()
    for name, s, e in [("2015-19", "2015", "2019"), ("2020-22", "2020", "2022"),
                       ("2023-26", "2023", "2026")]:
        print(f"    avg VIX {name}: {vm.loc[s:e].mean():.1f}")

    print("\n===== 5. IMPLEMENTABILITY (public-knowledge notes) =====")
    print("  - VXX is an ETN (Barclays credit). XIV (levered short-vol) TERMINATED "
          "Feb-2018; Barclays REDEEMED VXX itself in Jan-2019 and re-issued it. "
          "A redemption forces a cover at issuer's call price — you cannot 'hold "
          "8.5 years' a product the issuer can retire on 30 days notice.")
    print("  - Borrow: VXX is hard-to-borrow at retail brokers in calm periods and "
          "unborrowable/short-sale-restricted after spikes; brokers auto-buy-in "
          "after corporate actions (incl. the 5 reverse splits) and on lender "
          "recall — a recall after a spike = forced cover near the top.")
    print("  - Margin: Reg-T 50% initial, 30% NYSE maintenance for shorts "
          "(brokers typically require 40%): Feb-2018 (above) liquidates the "
          "position on day 1-2 of the spike — the backtest's recovery is fiction "
          "for a real account; you re-enter only at broker's discretion.")
    print("  - maxDD -67.57% monthly-close understates it: intraday the drawdown "
          "was worse, and no small account survives a -67% equity DD on a "
          "concentrated short-vol book without top-ups (which are unmodeled "
          "capital calls).")
    print("\nDONE")


if __name__ == "__main__":
    main()
