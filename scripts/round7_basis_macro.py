#!/usr/bin/env python3
"""Round 7: H19 premium contrarian, H21 DXY/BTC macro timing.

H19: Hyperliquid perp-spot premium (basis) as a contrarian signal. I have
~3.2y of 8h premium for BTC/ETH/SOL (results/cache/hl_funding_*.csv).
Design: each week, z = (premium - trailing-30d mean)/trailing-30d std.
If z > 1.5 (overheated longs) -> short BTC next week; z < -1.5 -> long;
else flat. BTC weekly returns from Coinbase daily. Costs 10bps/side.
Regimes in thirds. Baseline: BTC buy-hold.

H21: DXY vs BTC. DX-Y.NYB daily (yfinance). Rule: long BTC when DXY <
its 200d MA (weak dollar), else cash (T-bill ~0). Daily rebalance check,
costs on flips 10bps/side. 2021-2026 (BTC daily limit). Baseline BTC buy-hold.

H20 (FOMC drift): SKIPPED: no trustworthy machine-readable historical FOMC
calendar found in this loop; hardcoding dates from memory risks silent error.
Logged as future work needing a verified calendar source.

No lookahead: z-scores and MAs use strictly past data; weekly positions set
at week close, earn next week.
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

COST_CR = 0.001


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def h19_premium_contrarian():
    print("\n===== H19 HL premium contrarian (BTC) =====")
    f = pd.read_csv(BASE / "results" / "cache" / "hl_funding_BTC.csv",
                    index_col=0, parse_dates=True, date_format="mixed")
    f.index = pd.to_datetime(f.index, utc=True)
    f = f.sort_index()
    # weekly premium: last observation each week
    wp = f["premium"].resample("W").last().dropna()
    mu = wp.rolling(4).mean()   # ~30d
    sd = wp.rolling(4).std()
    z = (wp - mu) / sd
    btc = fetch_klines("BTCUSDT", "1d", total=1500)
    btc.index = pd.to_datetime(btc.index, utc=True)
    wr = np.log(btc["close"] / btc["close"].shift(1)).resample("W").sum().dropna()
    # align: signal at week t (premium through week t) -> position week t+1
    pos, rets = 0, []
    weeks = wp.index.intersection(wr.index)
    prev = 0
    for i, t in enumerate(weeks[:-1]):
        zv = z.loc[t]
        tgt = 0
        if zv > 1.5:
            tgt = -1
        elif zv < -1.5:
            tgt = 1
        t1 = weeks[i + 1]
        r = tgt * float(wr.loc[t1]) - (COST_CR * 2 if tgt != prev else 0.0)
        rets.append(r)
        prev = tgt
    pr = pd.Series(rets, index=weeks[1:len(rets) + 1])
    eq = np.exp(pr.cumsum())
    bh = np.exp(wr.loc[pr.index].cumsum())
    n = len(pr)
    thirds = [0, n // 3, 2 * n // 3, n]
    print(f"  n={n}w tot={eq.iloc[-1]-1:+.2%} Sharpe={ann_sharpe(pr,52):+.2f} maxDD={max_dd(eq):.2%}")
    for i, R in enumerate(["R1", "R2", "R3"]):
        s = pr.iloc[thirds[i]:thirds[i + 1]]
        print(f"    {R}: tot={np.exp(s.cumsum()).iloc[-1]-1:+.2%} Sh={ann_sharpe(s,52):+.2f}")
    print(f"  BTC buy-hold: tot={bh.iloc[-1]-1:+.2%} Sh={ann_sharpe(wr.loc[pr.index],52):+.2f}")
    print(f"  corr(premium_z, next-week BTC ret) = {z.shift(0).loc[weeks[:-1]].corr(wr.shift(-1).loc[weeks[:-1]]):+.3f} "
          "(contrarian => expect negative)")


def h21_dxy_btc():
    print("\n===== H21 DXY < 200d MA -> long BTC =====")
    dxy = yf.download("DX-Y.NYB", start="2021-01-01", end="2026-09-01",
                      interval="1d", auto_adjust=True, progress=False)
    if isinstance(dxy.columns, pd.MultiIndex):
        dxy.columns = dxy.columns.get_level_values(0)
    dxy = dxy["Close"].dropna()
    dxy.index = pd.to_datetime(dxy.index, utc=True)
    ma = dxy.rolling(200).mean()
    btc = fetch_klines("BTCUSDT", "1d", total=1500)
    btc.index = pd.to_datetime(btc.index, utc=True)
    br = np.log(btc["close"] / btc["close"].shift(1)).dropna()
    idx = dxy.index.intersection(br.index)
    sig = (dxy.loc[idx] < ma.loc[idx]).astype(float).shift(1)  # trade next day
    pr = (sig * br.loc[idx]).fillna(0)
    pr = pr - (sig.diff().abs() > 0).astype(float) * COST_CR * 2
    eq = np.exp(pr.cumsum())
    bh = np.exp(br.loc[idx].cumsum())
    mid = len(pr) // 2
    print(f"  n={len(pr)}d tot={eq.iloc[-1]-1:+.2%} Sharpe={ann_sharpe(pr,365):+.2f} maxDD={max_dd(eq):.2%} "
          f"time-in={(sig>0).mean():.0%}")
    print(f"    H1 Sh={ann_sharpe(pr.iloc[:mid],365):+.2f} H2 Sh={ann_sharpe(pr.iloc[mid:],365):+.2f}")
    print(f"  BTC buy-hold: tot={bh.iloc[-1]-1:+.2%} Sh={ann_sharpe(br.loc[idx],365):+.2f}")


if __name__ == "__main__":
    h19_premium_contrarian()
    h21_dxy_btc()
    print("\nDONE")
