#!/usr/bin/env python3
"""Round 10 (final): H26, using H4 (VIX>30 spike) as a satellite overlay on SPY.

The loop's sole survivor (H4) trades 2.6x/year, so it's economically negligible
standalone. The right question: does it add value ON TOP of buy-and-hold?
Portfolio: 100% SPY buy-hold + overlay that goes to 200% long SPY for 5
sessions after each fresh VIX>30 close (i.e., the H4 trade stacked on the
base position). Costs 5bps/side on the overlay leg only.
Compare: SPY buy-hold vs buy-hold+overlay, 2015-2026, per-regime Sharpe,
maxDD, and the overlay's standalone contribution. If the overlay doesn't
move the needle, H4 is an interesting statistical anomaly with no
portfolio use. That is the honest final verdict.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
COST = 0.0005


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def main():
    print("===== H26: H4 VIX-spike overlay on SPY buy-hold =====")
    spy = yf.download("SPY", start="2015-01-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    vix = yf.download("^VIX", start="2014-12-01", end="2026-09-01", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    # fresh VIX>30: close>30 and no close>30 in prior 20 sessions
    vc = vix["Close"]
    fresh = (vc > 30) & (vc.rolling(20, min_periods=1).max().shift(1) <= 30)
    # H4 trade: enter next open, hold 5 sessions. Approximate with closes:
    # overlay active on the 5 sessions after the signal day.
    rets = np.log(spy["Close"] / spy["Close"].shift(1))
    sig_dates = vc.index[fresh.fillna(False)]
    sig_dates = sig_dates[sig_dates >= "2015-01-01"]
    overlay = pd.Series(0.0, index=spy.index)
    for d in sig_dates:
        i = spy.index.searchsorted(d, side="right")
        for k in range(i, min(i + 5, len(spy))):
            overlay.iloc[k] = 1.0
    # portfolio: 1.0 base + 1.0 overlay when active
    base = rets.fillna(0)
    ov = (overlay * rets).fillna(0)
    ov_trades = int((overlay.diff().abs() > 0).sum() / 2)
    ov = ov - (overlay.diff().abs() > 0).astype(float) * COST
    port = base + ov
    for lab, pr, ppy in [("SPY buy-hold", base, 252), ("+ VIX overlay", port, 252),
                         ("overlay leg only", ov, 252)]:
        eq = np.exp(pr.cumsum())
        print(f"  {lab}: tot={eq.iloc[-1]-1:+.2%} Sh={ann_sharpe(pr,ppy):+.2f} maxDD={max_dd(eq):.2%}")
    print(f"  overlay: {len(sig_dates)} events, {ov_trades} round trips, "
          f"standalone tot={np.exp(ov.cumsum()).iloc[-1]-1:+.2%}")
    for name, s, e in [("2015-19", "2015", "2019"), ("2020-22", "2020", "2022"),
                       ("2023-26", "2023", "2026")]:
        b, p = base.loc[s:e], port.loc[s:e]
        print(f"    {name}: buy-hold Sh={ann_sharpe(b,252):+.2f} | +overlay Sh={ann_sharpe(p,252):+.2f}")


if __name__ == "__main__":
    main()
    print("\nDONE")
