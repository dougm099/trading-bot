#!/usr/bin/env python3
"""Round 5: H14 weekly short-term reversal (large caps), H15 dual momentum.

H14: Jegadeesh/Lehmann weekly reversal on ~40 large-cap US stocks with full
2015+ history (survivorship bias noted as caveat). Each week: rank by prior
5-trading-day return; long bottom quintile, short top quintile, equal weight,
hold 1 week (Friday close -> Friday close). Variants: standard (full weekly
rebalance) and "smart" low-turnover (hold incumbent positions unless they
leave the outer tercile). Costs 5bps/side. Regimes: 2015-19 / 2020-22 / 2023-26.
Baseline: SPY buy-hold.

H15: Antonacci dual momentum, monthly: universe SPY (US), EFA (intl), AGG
(bonds). Absolute filter: 12m total return > 0 (else cash). Among passing,
hold the one with highest 12m return. Rebalance monthly. Costs 5bps/side.
Baseline: 60/40 SPY/AGG buy-hold and SPY buy-hold. 2015-2026.

No lookahead: H14 ranks on completed weeks, trades at Friday close; H15 uses
month-end closes only.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
COST = 0.0005

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
            "BRK-B", "JPM", "V", "JNJ", "XOM", "WMT", "MA", "PG", "ORCL",
            "COST", "HD", "NFLX", "BAC", "ABBV", "CRM", "AMD", "KO", "PEP",
            "TMO", "DIS", "INTC", "VZ", "ADBE", "CSCO", "PFE", "MCD", "ABT",
            "QCOM", "TXN", "HON", "AMGN", "IBM"]


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def h14_weekly_reversal():
    print("\n===== H14 weekly short-term reversal (large caps) =====")
    print("  CAVEAT: fixed 40-stock universe = survivorship bias; results are upper-bound.")
    px = yf.download(UNIVERSE, start="2015-01-01", end="2026-09-01",
                     interval="1d", auto_adjust=True, progress=False)["Close"].dropna(how="all")
    px = px.dropna(axis=1)
    print(f"  {px.shape[1]} stocks, {px.index[0].date()} -> {px.index[-1].date()}")
    # weekly bars Friday-close
    w = px.resample("W-FRI").last()
    wr = np.log(w / w.shift(1))  # weekly log returns
    # signal week t -> hold week t+1
    n = len(w)
    results = {}
    for variant in ["standard", "smart"]:
        port_rets, turnover = [], 0.0
        prev_pos = pd.Series(0.0, index=w.columns)
        for t in range(1, n - 1):
            r_t = wr.iloc[t].dropna()
            if len(r_t) < 20:
                port_rets.append(0.0); continue
            q = r_t.quantile([0.2, 0.8])
            target = pd.Series(0.0, index=w.columns)
            losers = r_t[r_t <= q.iloc[0]].index
            winners = r_t[r_t >= q.iloc[1]].index
            if len(losers) and len(winners):
                target[losers] = 1.0 / len(losers)
                target[winners] = -1.0 / len(winners)
            if variant == "smart":
                # keep incumbents unless they leave outer terciles
                q3 = r_t.quantile([1 / 3, 2 / 3])
                keep_long = prev_pos[prev_pos > 0].index[r_t.reindex(prev_pos[prev_pos > 0].index) <= q3.iloc[1]]
                keep_short = prev_pos[prev_pos < 0].index[r_t.reindex(prev_pos[prev_pos < 0].index) >= q3.iloc[0]]
                kept = pd.Series(0.0, index=w.columns)
                if len(keep_long):
                    kept[keep_long] = 1.0 / max(len(losers), 1)
                if len(keep_short):
                    kept[keep_short] = -1.0 / max(len(winners), 1)
                # fill remainder with new names
                new_long = [s for s in losers if s not in keep_long][: max(0, len(losers) - len(keep_long))]
                new_short = [s for s in winners if s not in keep_short][: max(0, len(winners) - len(keep_short))]
                if new_long:
                    kept[new_long] = 1.0 / max(len(losers), 1)
                if new_short:
                    kept[new_short] = -1.0 / max(len(winners), 1)
                # normalize to $1 long / $1 short
                lp, sp = kept[kept > 0].sum(), -kept[kept < 0].sum()
                if lp > 0:
                    kept[kept > 0] /= lp
                if sp > 0:
                    kept[kept < 0] /= sp
                target = kept
            dw = (target - prev_pos).abs().sum() / 2  # dollars traded per $1 gross
            turnover += dw
            r_next = wr.iloc[t + 1].reindex(w.columns).fillna(0)
            port_rets.append(float((target * r_next).sum()) - dw * COST * 2)
            prev_pos = target
        pr = pd.Series(port_rets, index=w.index[2:2 + len(port_rets)])
        eq = np.exp(pr.cumsum())
        results[variant] = pr
        print(f"  {variant}: n={len(pr)}w tot={eq.iloc[-1]-1:+.2%} Sharpe={ann_sharpe(pr,52):+.2f} "
              f"maxDD={max_dd(eq):.2%} turnover~{turnover:.0f}x")
        for name, s, e in [("2015-19", "2015", "2019"), ("2020-22", "2020", "2022"),
                           ("2023-26", "2023", "2026")]:
            sub = pr.loc[s:e]
            print(f"    {name}: tot={np.exp(sub.cumsum()).iloc[-1]-1:+.2%} Sh={ann_sharpe(sub,52):+.2f}")
    spy = yf.download("SPY", start="2015-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    bhr = np.log(spy["Close"] / spy["Close"].shift(1)).resample("W-FRI").sum()
    print(f"  SPY buy-hold weekly: tot={np.exp(bhr.cumsum()).iloc[-1]-1:+.2%} "
          f"Sh={ann_sharpe(bhr.dropna(),52):+.2f}")


def h15_dual_momentum():
    print("\n===== H15 dual momentum (SPY/EFA/AGG, monthly) =====")
    px = yf.download(["SPY", "EFA", "AGG"], start="2014-01-01", end="2026-09-01",
                     interval="1d", auto_adjust=True, progress=False)["Close"]
    m = px.resample("M").last()
    mr = m.pct_change(1)
    mom12 = m.pct_change(12)
    port = []
    prev = None
    for t in m.index[12:-1]:
        cands = mom12.loc[t].dropna()
        passing = cands[cands > 0]
        pick = passing.idxmax() if len(passing) else "CASH"
        nxt = mr.loc[m.index[m.index.get_loc(t) + 1]]
        r = 0.0 if pick == "CASH" else float(nxt[pick])
        cost = COST * 2 if pick != prev else 0.0
        port.append(r - cost)
        prev = pick
    pr = pd.Series(port, index=m.index[13:13 + len(port)])
    eq = (1 + pr).cumprod()
    # baselines
    bh6040 = (0.6 * mr["SPY"] + 0.4 * mr["AGG"]).loc[pr.index]
    eqb = (1 + bh6040).cumprod()
    spybh = mr["SPY"].loc[pr.index]
    eqs = (1 + spybh).cumprod()
    print(f"  dual-mom: n={len(pr)}m tot={eq.iloc[-1]-1:+.2%} Sharpe={ann_sharpe(np.log1p(pr),12):+.2f} "
          f"maxDD={max_dd(eq):.2%}")
    for name, s, e in [("2015-19", "2015", "2019"), ("2020-22", "2020", "2022"),
                       ("2023-26", "2023", "2026")]:
        sub = pr.loc[s:e]
        print(f"    {name}: tot={(1+sub).cumprod().iloc[-1]-1:+.2%} Sh={ann_sharpe(np.log1p(sub),12):+.2f}")
    print(f"  60/40 baseline: tot={eqb.iloc[-1]-1:+.2%} Sh={ann_sharpe(np.log1p(bh6040),12):+.2f}")
    print(f"  SPY buy-hold: tot={eqs.iloc[-1]-1:+.2%} Sh={ann_sharpe(np.log1p(spybh),12):+.2f}")


if __name__ == "__main__":
    h14_weekly_reversal()
    h15_dual_momentum()
    print("\nDONE")
