#!/usr/bin/env python3
"""Round 4: classic quant algos continued.

H11: Donchian/Turtle volatility breakout. Long when close > max(high, N),
exit when close < min(low, M). N in {20, 55}, M = N//2. Test on BTC daily
(Coinbase, ~4y) and SPY daily (yfinance, 2015-2026). Long-only + long/short
variants. Costs: 10bps/side crypto, 5bps equities. Per-regime.

H12: cross-sectional momentum on crypto. Universe: BTC, ETH, SOL (+ BNB, XRP,
DOGE, ADA, AVAX via Coinbase daily). Monthly rebalance: long top-2 by
trailing 30d return, equal weight, hold 1 month. 10bps/side. Baseline:
equal-weight buy-and-hold of universe. Regimes.

H13: BTC time-series momentum (Moskowitz 12-1): sign of trailing 12m return
(minus last month) -> long/short BTC monthly. Daily Coinbase data 2021-2026.
10bps/side.

No lookahead: all signals use strictly past closes.
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


def h11_donchian():
    print("\n===== H11 Donchian breakout =====")
    # BTC daily
    btc = fetch_klines("BTCUSDT", "1d", total=1500)
    btc.index = pd.to_datetime(btc.index, utc=True)
    spy = yf.download("SPY", start="2015-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    for name, df, cost, ppy in [("BTC", btc, COST_CR, 365), ("SPY", spy, COST_EQ, 252)]:
        df = df.copy()
        df["ret"] = np.log(df["Close"] / df["Close"].shift(1)) if "Close" in df else \
            np.log(df["close"] / df["close"].shift(1))
        hi = df["High"] if "High" in df else df["high"]
        lo = df["Low"] if "Low" in df else df["low"]
        cl = df["Close"] if "Close" in df else df["close"]
        for N in [20, 55]:
            M = N // 2
            up, dn = hi.rolling(N).max().shift(1), lo.rolling(M).min().shift(1)
            pos_lo = pd.Series(0, index=df.index)
            p = 0
            for i in range(len(df)):
                if p == 0 and cl.iloc[i] > up.iloc[i]:
                    p = 1
                elif p == 1 and cl.iloc[i] < dn.iloc[i]:
                    p = 0
                pos_lo.iloc[i] = p
            r = (pos_lo.shift(1) * df["ret"]).fillna(0)
            tr = (pos_lo.diff().abs() > 0).sum() / 2
            r_net = r - (pos_lo.diff().abs() > 0).astype(float) * cost
            eq = np.exp(r_net.cumsum())
            mid = len(r) // 2
            print(f"  {name} N={N} long-only: trades~{int(tr)} tot={eq.iloc[-1]-1:+.2%} "
                  f"Sh={ann_sharpe(r_net,ppy):+.2f} maxDD={max_dd(eq):.2%} "
                  f"| H1 Sh={ann_sharpe(r_net.iloc[:mid],ppy):+.2f} H2 Sh={ann_sharpe(r_net.iloc[mid:],ppy):+.2f}")
        # SPY long/short variant N=20
        if name == "SPY":
            N, M = 20, 10
            up, dn = hi.rolling(N).max().shift(1), lo.rolling(M).min().shift(1)
            pos_ls = pd.Series(0, index=df.index)
            p = 0
            for i in range(len(df)):
                if cl.iloc[i] > up.iloc[i]:
                    p = 1
                elif cl.iloc[i] < dn.iloc[i]:
                    p = -1
                pos_ls.iloc[i] = p
            r = (pos_ls.shift(1) * df["ret"]).fillna(0)
            r_net = r - (pos_ls.diff().abs() > 0).astype(float) * cost
            eq = np.exp(r_net.cumsum())
            print(f"  SPY N=20 long/short: tot={eq.iloc[-1]-1:+.2%} Sh={ann_sharpe(r_net,ppy):+.2f} "
                  f"maxDD={max_dd(eq):.2%}")


def h12_xsmom_crypto():
    print("\n===== H12 crypto cross-sectional momentum =====")
    uni = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX"]  # BNB not on Coinbase
    px = {}
    for c in uni:
        try:
            b = fetch_klines(f"{c}USDT", "1d", total=1500)
            px[c] = b["close"]
        except Exception as e:
            print(f"  {c}: {e}")
    px = pd.DataFrame(px).dropna()
    px.index = pd.to_datetime(px.index, utc=True)
    print(f"  universe {list(px.columns)}, {px.index[0].date()} -> {px.index[-1].date()}")
    m = px.resample("M").last()          # month-end closes
    mr = m.pct_change(1)                 # monthly returns, known at month end
    port, bh = [], []
    # at month-end t (t>=12): rank by trailing 1m return through t, hold top2 over next month
    for k in range(12, len(m) - 1):
        t = m.index[k]
        mom = mr.loc[t].dropna()
        if len(mom) < 3:
            continue
        top2 = mom.nlargest(2).index.tolist()
        nxt_ret = mr.iloc[k + 1]
        port.append(nxt_ret[top2].mean() - COST_CR * 2)  # full turnover each month
        bh.append(nxt_ret.mean())
    port = pd.Series(port, index=m.index[13:13 + len(port)])
    bh = pd.Series(bh, index=port.index)
    eq, eqb = (1 + port).cumprod(), (1 + bh).cumprod()
    mid = len(port) // 2
    print(f"  top2-momentum: n={len(port)}m tot={eq.iloc[-1]-1:+.2%} Sh={ann_sharpe(np.log1p(port),12):+.2f} "
          f"maxDD={max_dd(eq):.2%} | H1 Sh={ann_sharpe(np.log1p(port.iloc[:mid]),12):+.2f} "
          f"H2 Sh={ann_sharpe(np.log1p(port.iloc[mid:]),12):+.2f}")
    print(f"  equal-weight buy-hold: tot={eqb.iloc[-1]-1:+.2%} Sh={ann_sharpe(np.log1p(bh),12):+.2f}")


def h13_tsmom_btc():
    print("\n===== H13 BTC 12-1 time-series momentum =====")
    btc = fetch_klines("BTCUSDT", "1d", total=2200)
    btc.index = pd.to_datetime(btc.index, utc=True)
    m = btc["close"].resample("M").last()
    r = np.log(m / m.shift(1))
    mom = (m.shift(1) / m.shift(12) - 1)  # 12-1
    sig = np.sign(mom).shift(1)  # trade next month
    flips = (sig.diff().abs() > 0).fillna(False)
    pr = (sig * r).dropna()
    pr = pr - flips.reindex(pr.index).fillna(False).astype(float) * COST_CR * 2
    eq = np.exp(pr.cumsum())
    bh = np.exp(r.loc[pr.index].cumsum())
    mid = len(pr) // 2
    print(f"  n={len(pr)}m tot={eq.iloc[-1]-1:+.2%} Sh={ann_sharpe(pr,12):+.2f} maxDD={max_dd(eq):.2%} "
          f"| H1 Sh={ann_sharpe(pr.iloc[:mid],12):+.2f} H2 Sh={ann_sharpe(pr.iloc[mid:],12):+.2f}")
    print(f"  BTC buy-hold: tot={bh.iloc[-1]-1:+.2%} Sh={ann_sharpe(r.loc[pr.index],12):+.2f}")


if __name__ == "__main__":
    h11_donchian()
    h12_xsmom_crypto()
    h13_tsmom_btc()
    print("\nDONE")
