#!/usr/bin/env python3
"""Round 8: H22 low-volatility anomaly, H23 52-week-high momentum.

Universe: the same 40 large caps as H14 (survivorship caveat applies;
results are upper bounds). Monthly rebalance, 5bps/side.

H22 (betting-against-beta / low-vol): each month-end, rank by trailing 60d
realized vol; long bottom quintile, short top quintile, equal weight, hold
1 month.

H23 (George & Hwang 2004): each month-end, rank by proximity to 52-week
high (close / trailing-252d max); long top quintile (nearest highs), short
bottom quintile, hold 1 month.

Regimes: 2015-19 / 2020-22 / 2023-26. Baselines: SPY buy-hold, and for H22
also the long-only low-vol leg alone.
No lookahead: ranks use data through month-end, positions earn next month.
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


def load():
    px = yf.download(UNIVERSE, start="2015-01-01", end="2026-09-01",
                     interval="1d", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(axis=1)
    print(f"  {px.shape[1]} stocks, {px.index[0].date()} -> {px.index[-1].date()}")
    return px


def backtest_ls(px, rank_fn, label):
    m = px.resample("M").last()
    mr = np.log(m / m.shift(1))
    dr = np.log(px / px.shift(1))
    port, long_only = [], []
    for k in range(12, len(m) - 1):
        t = m.index[k]
        ranks = rank_fn(px.loc[:t], t)
        ranks = ranks.dropna()
        if len(ranks) < 20:
            port.append(0.0); long_only.append(0.0); continue
        q = ranks.quantile([0.2, 0.8])
        longs = ranks[ranks <= q.iloc[0]].index
        shorts = ranks[ranks >= q.iloc[1]].index
        nxt = mr.iloc[k + 1]
        rl = float(nxt.reindex(longs).mean()) if len(longs) else 0.0
        rs = float(nxt.reindex(shorts).mean()) if len(shorts) else 0.0
        port.append(0.5 * rl - 0.5 * rs - COST * 2)
        long_only.append(rl - COST * 2)
    pr = pd.Series(port, index=m.index[13:13 + len(port)])
    lo = pd.Series(long_only, index=pr.index)
    eq, eql = np.exp(pr.cumsum()), np.exp(lo.cumsum())
    print(f"  {label} L/S: n={len(pr)}m tot={eq.iloc[-1]-1:+.2%} Sh={ann_sharpe(pr,12):+.2f} "
          f"maxDD={max_dd(eq):.2%}")
    print(f"  {label} long-only: tot={eql.iloc[-1]-1:+.2%} Sh={ann_sharpe(lo,12):+.2f} "
          f"maxDD={max_dd(eql):.2%}")
    for name, s, e in [("2015-19", "2015", "2019"), ("2020-22", "2020", "2022"),
                       ("2023-26", "2023", "2026")]:
        sub = pr.loc[s:e]
        print(f"    L/S {name}: tot={np.exp(sub.cumsum()).iloc[-1]-1:+.2%} Sh={ann_sharpe(sub,12):+.2f}")
    return pr


if __name__ == "__main__":
    print("===== H22 low-volatility anomaly =====")
    print("  CAVEAT: fixed 40-stock universe = survivorship bias; upper bound.")
    px = load()

    def vol_rank(hist, t):
        return hist.pct_change().iloc[-60:].std()

    backtest_ls(px, vol_rank, "low-vol")

    print("\n===== H23 52-week-high momentum =====")

    def hw52_rank(hist, t):
        return -(hist.iloc[-252:].iloc[-1] / hist.iloc[-252:].max())  # lower = nearer high

    backtest_ls(px, hw52_rank, "52w-high")
    spy = yf.download("SPY", start="2016-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    mr = np.log(spy["Close"] / spy["Close"].shift(1)).resample("M").sum().dropna()
    print(f"  SPY buy-hold monthly: tot={np.exp(mr.cumsum()).iloc[-1]-1:+.2%} Sh={ann_sharpe(mr,12):+.2f}")
    print("\nDONE")
