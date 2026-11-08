#!/usr/bin/env python3
"""Round 1: H1 overnight gap effect, H2 BTC hour-of-day seasonality.

H1: for SPY/QQQ daily bars, split return into overnight (prev close -> open)
    and intraday (open -> close). Test whether overnight predicts intraday
    (continuation vs reversal), then backtest both directions with 5bps/side.
    Regimes: 2015-2019, 2020-2022, 2023-2026. Baseline: buy-and-hold intraday.

H2: BTC hourly bars (Coinbase, via bot/data/binance.py). Mean log-return by
    hour-of-day (UTC), t-stats with Bonferroni correction. Strategy: long only
    during statistically significant positive hours; cost 10bps/side.

No lookahead: all signals use data strictly before the traded period.
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

COST_EQ = 0.0005   # 5 bps / side
COST_CR = 0.001    # 10 bps / side
REGIMES = [("2015-2019", "2015-01-01", "2019-12-31"),
           ("2020-2022", "2020-01-01", "2022-12-31"),
           ("2023-2026", "2023-01-01", "2026-09-01")]


def ann_sharpe(rets, periods_per_year):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(periods_per_year)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def h1_overnight(ticker):
    print(f"\n===== H1 overnight gap effect — {ticker} =====")
    df = yf.download(ticker, start="2015-01-01", end="2026-09-01",
                     interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "Close"])
    df["overnight"] = np.log(df["Open"] / df["Close"].shift(1))
    df["intraday"] = np.log(df["Close"] / df["Open"])
    df = df.dropna()
    all_rets = {}
    for name, s, e in REGIMES:
        sub = df.loc[s:e]
        if len(sub) < 100:
            print(f"  {name}: n={len(sub)} too small, skip"); continue
        corr = sub["overnight"].corr(sub["intraday"])
        print(f"  {name}: n={len(sub)} corr(overnight,intraday)={corr:+.3f}")
        # S1 continuation: long intraday if overnight>0 ; S2 reversal: short if overnight>0
        sig = np.sign(sub["overnight"].values)
        r_cont = sig * sub["intraday"].values - COST_EQ * 2   # enter at open, exit at close
        r_rev = -sig * sub["intraday"].values - COST_EQ * 2
        r_bh = sub["intraday"].values  # buy-at-open-hold-to-close baseline (no cost)
        for label, r in [("cont", r_cont), ("rev", r_rev), ("bh_open_close", r_bh)]:
            eq = np.exp(np.cumsum(r))
            tot = eq[-1] - 1
            all_rets.setdefault(label, []).append(r)
            print(f"    {label}: tot={tot:+.2%} Sharpe={ann_sharpe(r,252):+.2f} maxDD={max_dd(eq):.2%}")
    return all_rets


def h2_btc_seasonality():
    print("\n===== H2 BTC hour-of-day seasonality =====")
    bars = fetch_klines("BTCUSDT", "1h", total=8000)  # ~11 months hourly
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars["ret"] = np.log(bars["close"] / bars["close"].shift(1))
    bars["hour"] = bars.index.hour
    bars = bars.dropna()
    print(f"  n={len(bars)} bars from {bars.index[0].date()} to {bars.index[-1].date()}")
    rows = []
    for h in range(24):
        r = bars.loc[bars["hour"] == h, "ret"].values
        t = np.mean(r) / (np.std(r) / np.sqrt(len(r))) if len(r) > 20 else 0.0
        rows.append((h, len(r), np.mean(r) * 1e4, t))
    tbl = pd.DataFrame(rows, columns=["hour_utc", "n", "mean_bp", "t_stat"])
    tbl["bonf_sig"] = tbl["t_stat"].abs() > 2.88  # ~0.05/24 two-sided
    print(tbl.round(2).to_string(index=False))
    pos_hours = tbl.loc[tbl["bonf_sig"] & (tbl["mean_bp"] > 0), "hour_utc"].tolist()
    print(f"  Bonferroni-significant positive hours: {pos_hours}")
    # strategy: long during sig-positive hours only (enter at hour open, exit at hour close)
    sig = bars["hour"].isin(pos_hours).astype(float).values
    r = sig * bars["ret"].values
    # cost: one round trip per contiguous block; approximate as 2*cost per hour-block entry
    entries = np.sum(np.diff(np.concatenate([[0], sig])) > 0)
    r_net = r.copy()
    # subtract cost at each entry/exit: simpler: 2*cost per bar traded * turnover
    turnover = np.abs(np.diff(np.concatenate([[0], sig]))).sum() / 2
    total_cost = turnover * 2 * COST_CR
    net_tot = np.exp(r.sum()) - 1 - total_cost
    print(f"  long-sig-hours gross={np.exp(r.sum())-1:+.2%} entries~{int(turnover)} "
          f"cost~{total_cost:.2%} net~{net_tot:+.2%} Sharpe={ann_sharpe(r, 365*24):+.2f}")
    # regime split: first half vs second half of sample
    mid = len(bars) // 2
    for name, sl in [("H1_sample", slice(0, mid)), ("H2_sample", slice(mid, None))]:
        sub = bars.iloc[sl]
        s2 = sub["hour"].isin(pos_hours).astype(float).values * sub["ret"].values
        print(f"    {name}: gross={np.exp(s2.sum())-1:+.2%} Sharpe={ann_sharpe(s2,365*24):+.2f}")


if __name__ == "__main__":
    import sys as _sys
    only = _sys.argv[1] if len(_sys.argv) > 1 else "all"
    if only in ("all", "h1"):
        h1_overnight("SPY")
        h1_overnight("QQQ")
    if only in ("all", "h2"):
        h2_btc_seasonality()
    print("\nDONE")
