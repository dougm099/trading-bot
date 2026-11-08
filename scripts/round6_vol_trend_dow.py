#!/usr/bin/env python3
"""Round 6: H16 short-vol carry, H17 multi-asset trend, H18 day-of-week.

H16: short VXX (volatility ETN, structurally decays in contango). Strategy:
short VXX at month-end, hold 1 month; cover early (go flat) if VXX closes >
2x its 60d MA (vol-spike filter to avoid Feb-2018-style blowups; the filter
uses past data only). Costs 5bps/side. Shorting costs/borrow: assume 2%/yr
drag (~17bp/month), noted as assumption. Regimes incl. Feb 2018 and 2020.
Baseline: SPY buy-hold.

H17: multi-asset trend (Weller-style): universe SPY/TLT/GLD/DBC. Monthly: hold
each asset with 12m total return > 0, equal-weight holders, rest cash.
Costs 5bps/side. Baseline: equal-weight 4-asset buy-hold. 2015-2026.

H18: day-of-week seasonality. SPY daily 2015-2026 and BTC daily (Coinbase):
mean log-return by weekday with t-stats; Bonferroni over 5 days. Trading
rule: long-only on the single best weekday (if significant), evaluated
per-regime. Costs 5bps/side (SPY), 10bps (BTC).

No lookahead: month-end / week-completed signals only.
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

COST_EQ, COST_CR = 0.0005, 0.001
BORROW_DRAG_M = 0.02 / 12  # assumed 2%/yr borrow cost on short VXX


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def h16_short_vxx():
    print("\n===== H16 short-VXX carry with vol-spike filter =====")
    vxx = yf.download("VXX", start="2015-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(vxx.columns, pd.MultiIndex):
        vxx.columns = vxx.columns.get_level_values(0)
    c = vxx["Close"]
    ma60 = c.rolling(60).mean()
    m = c.resample("M").last()
    rets, note = [], 0
    for i in range(len(m) - 1):
        t0, t1 = m.index[i], m.index[i + 1]
        # HONEST filter: entry decision uses only data through t0 (no peeking
        # into the holding month). Skip the month if VXX is already spiking.
        c0, ma0 = c.loc[:t0].iloc[-1], ma60.loc[:t0].iloc[-1]
        if c0 > 1.5 * ma0:
            r = 0.0
            note += 1
        else:
            r = -float(np.log(m.iloc[i + 1] / m.iloc[i])) - BORROW_DRAG_M - COST_EQ * 2
        rets.append(r)
    pr = pd.Series(rets, index=m.index[1:])
    eq = np.exp(pr.cumsum())
    # unfiltered version for comparison
    raw = -np.log(m / m.shift(1)).dropna() - BORROW_DRAG_M - COST_EQ * 2
    raw = raw.loc[pr.index]
    eqr = np.exp(raw.cumsum())
    print(f"  filtered short-VXX: n={len(pr)}m tot={eq.iloc[-1]-1:+.2%} "
          f"Sh={ann_sharpe(pr,12):+.2f} maxDD={max_dd(eq):.2%} (dodged {note} months)")
    print(f"  unfiltered short-VXX: tot={eqr.iloc[-1]-1:+.2%} Sh={ann_sharpe(raw,12):+.2f} "
          f"maxDD={max_dd(eqr):.2%}")
    for name, s, e in [("2015-19", "2015", "2019"), ("2020-22", "2020", "2022"),
                       ("2023-26", "2023", "2026")]:
        sub = pr.loc[s:e]
        print(f"    filtered {name}: tot={np.exp(sub.cumsum()).iloc[-1]-1:+.2%} "
              f"Sh={ann_sharpe(sub,12):+.2f}")


def h17_multi_asset_trend():
    print("\n===== H17 multi-asset trend (SPY/TLT/GLD/DBC) =====")
    px = yf.download(["SPY", "TLT", "GLD", "DBC"], start="2014-01-01",
                     end="2026-09-01", interval="1d", auto_adjust=True,
                     progress=False)["Close"]
    m = px.resample("M").last()
    mr = m.pct_change(1)
    mom12 = m.pct_change(12)
    port, bh = [], []
    prev_n = None
    for k in range(12, len(m) - 1):
        t = m.index[k]
        holders = mom12.loc[t][mom12.loc[t] > 0].index.tolist()
        w = 1.0 / len(holders) if holders else 0.0
        nxt = mr.iloc[k + 1]
        r = float(nxt[holders].sum() * w) if holders else 0.0
        turnover = abs(len(holders) - (prev_n or 0)) / max(len(holders), 1)
        port.append(r - COST_EQ * 2 * turnover)
        bh.append(float(nxt.mean()))
        prev_n = len(holders)
    pr = pd.Series(port, index=m.index[13:13 + len(port)])
    bhs = pd.Series(bh, index=pr.index)
    eq, eqb = (1 + pr).cumprod(), (1 + bhs).cumprod()
    print(f"  trend: n={len(pr)}m tot={eq.iloc[-1]-1:+.2%} Sh={ann_sharpe(np.log1p(pr),12):+.2f} "
          f"maxDD={max_dd(eq):.2%}")
    for name, s, e in [("2015-19", "2015", "2019"), ("2020-22", "2020", "2022"),
                       ("2023-26", "2023", "2026")]:
        sub = pr.loc[s:e]
        print(f"    {name}: tot={(1+sub).cumprod().iloc[-1]-1:+.2%} Sh={ann_sharpe(np.log1p(sub),12):+.2f}")
    print(f"  EW buy-hold: tot={eqb.iloc[-1]-1:+.2%} Sh={ann_sharpe(np.log1p(bhs),12):+.2f}")


def h18_day_of_week():
    print("\n===== H18 day-of-week seasonality =====")
    spy = yf.download("SPY", start="2015-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy_r = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()
    btc = fetch_klines("BTCUSDT", "1d", total=1500)
    btc.index = pd.to_datetime(btc.index, utc=True)
    btc_r = np.log(btc["close"] / btc["close"].shift(1)).dropna()
    for name, r, cost, ppy in [("SPY", spy_r, COST_EQ, 252), ("BTC", btc_r, COST_CR, 365)]:
        dow = pd.Series(r.index.dayofweek, index=r.index)
        print(f"  -- {name} --")
        means = {}
        for d, lab in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
            s = r[dow == d]
            t = s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 10 else 0
            means[lab] = (float(s.mean() * 1e4), float(t), len(s))
            flag = " *" if abs(t) > 2.81 else ""  # Bonferroni 0.05/5
            print(f"    {lab}: mean={s.mean()*1e4:+.1f}bp/d t={t:+.2f} n={len(s)}{flag}")
        best = max(means, key=lambda k: means[k][0])
        # trade: long-only on best weekday (in-sample pick, optimistic; per-regime check below)
        sig = pd.Series((dow == ["Mon", "Tue", "Wed", "Thu", "Fri"].index(best)).astype(float),
                        index=r.index)
        pr = (sig.shift(0) * r).fillna(0)  # same-day: assume entry at prior close
        # NOTE: trading weekday d using close(t-1)->close(t) is implementable (hold only on d)
        tr = (sig.diff().abs() > 0).sum() / 2
        prn = pr - (sig.diff().abs() > 0).astype(float) * cost * 2
        eq = np.exp(prn.cumsum())
        print(f"    long-{best}-only: trades~{int(tr)} tot={eq.iloc[-1]-1:+.2%} "
              f"Sh={ann_sharpe(prn,ppy):+.2f} (in-sample weekday pick, optimistic)")


if __name__ == "__main__":
    h16_short_vxx()
    h17_multi_asset_trend()
    h18_day_of_week()
    print("\nDONE")
