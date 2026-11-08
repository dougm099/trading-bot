#!/usr/bin/env python3
"""Round 9: H24 crypto weekend effect, H25 safe-haven (gold) trend.

H24: BTC trades 24/7. Do weekends differ? Daily BTC (Coinbase): mean
log-return for Sat/Sun vs weekdays, t-stats with Bonferroni. Trading rule:
long-only weekends (Fri close -> Sun close) if significant; per-regime check.
Costs 10bps/side.

H25: gold trend-following (safe-haven): GLD daily (yfinance 2015-2026).
Rule: long GLD when close > 200d MA, else cash. Costs 5bps/side. Baseline:
GLD buy-hold. Per-regime. (Tests whether simple trend adds value on a
safe-haven asset rather than equity beta.)

No lookahead: weekend rule uses calendar only; MA uses past closes.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from bot.data.binance import fetch_klines  # noqa: E402

COST_CR, COST_EQ = 0.001, 0.0005


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def h24_weekend():
    print("\n===== H24 BTC weekend effect =====")
    btc = fetch_klines("BTCUSDT", "1d", total=1500)
    btc.index = pd.to_datetime(btc.index, utc=True)
    r = np.log(btc["close"] / btc["close"].shift(1)).dropna()
    dow = r.index.dayofweek
    is_we = dow.isin([5, 6])
    for lab, s in [("weekend", r[is_we]), ("weekday", r[~is_we])]:
        t = s.mean() / (s.std() / np.sqrt(len(s)))
        print(f"  {lab}: mean={s.mean()*1e4:+.1f}bp/d t={t:+.2f} n={len(s)}")
    # trade: long Fri close -> Sun close (weekend only)
    sig = pd.Series(is_we.astype(float), index=r.index)
    pr = (sig * r).fillna(0)
    prn = pr - (sig.diff().abs() > 0).astype(float) * COST_CR * 2
    eq = np.exp(prn.cumsum())
    bh = np.exp(r.cumsum())
    mid = len(pr) // 2
    print(f"  long-weekends-only: tot={eq.iloc[-1]-1:+.2%} Sh={ann_sharpe(prn,365):+.2f} maxDD={max_dd(eq):.2%} "
          f"| H1 Sh={ann_sharpe(prn.iloc[:mid],365):+.2f} H2 Sh={ann_sharpe(prn.iloc[mid:],365):+.2f}")
    print(f"  BTC buy-hold: tot={bh.iloc[-1]-1:+.2%} Sh={ann_sharpe(r,365):+.2f}")


def h25_gold_trend():
    print("\n===== H25 GLD 200d-MA trend =====")
    gld = yf.download("GLD", start="2015-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(gld.columns, pd.MultiIndex):
        gld.columns = gld.columns.get_level_values(0)
    c = gld["Close"].dropna()
    r = np.log(c / c.shift(1))
    ma = c.rolling(200).mean()
    sig = (c > ma).astype(float).shift(1)
    pr = (sig * r).fillna(0)
    prn = pr - (sig.diff().abs() > 0).astype(float) * COST_EQ * 2
    eq = np.exp(prn.cumsum())
    bh = np.exp(r.loc[pr.index].cumsum())
    print(f"  n={len(pr)}d tot={eq.iloc[-1]-1:+.2%} Sh={ann_sharpe(prn,252):+.2f} maxDD={max_dd(eq):.2%} "
          f"time-in={(sig>0).mean():.0%}")
    for name, s, e in [("2015-19", "2015", "2019"), ("2020-22", "2020", "2022"),
                       ("2023-26", "2023", "2026")]:
        sub = prn.loc[s:e]
        print(f"    {name}: tot={np.exp(sub.cumsum()).iloc[-1]-1:+.2%} Sh={ann_sharpe(sub,252):+.2f}")
    print(f"  GLD buy-hold: tot={bh.iloc[-1]-1:+.2%} Sh={ann_sharpe(r.loc[pr.index],252):+.2f}")


if __name__ == "__main__":
    h24_weekend()
    h25_gold_trend()
    print("\nDONE")
