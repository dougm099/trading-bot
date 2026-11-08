#!/usr/bin/env python3
"""Backtest the experimental VIX fear-spike strategy on SPY daily bars.

Fetches SPY daily + ^VIX/^VIX3M via yfinance, builds the VIX3M/VIX slope
series, and runs it through the standard backtest harness (next-open fills,
risk manager sizing/guards). Research-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.backtest import run_backtest
from bot.config import Config
from bot.strategies.vix_fear import VixFearReversion


def main() -> None:
    raw = yf.download(["SPY", "^VIX", "^VIX3M"], period="5y", interval="1d",
                      progress=False, auto_adjust=True)
    close = raw["Close"]
    bars = pd.DataFrame({
        "open": raw["Open"]["SPY"],
        "high": raw["High"]["SPY"],
        "low": raw["Low"]["SPY"],
        "close": close["SPY"],
        "volume": raw["Volume"]["SPY"],
    }).dropna()
    slope = (close["^VIX3M"] / close["^VIX"]).dropna()
    print(f"SPY daily bars: {len(bars)} ({bars.index[0].date()} -> {bars.index[-1].date()})")
    print(f"VIX slope days: {len(slope)}")

    strat = VixFearReversion(vix_slope=slope, hold_days=20, threshold=1.0)
    res = run_backtest(strat, "SPY", bars, Config(), allow_short=False)

    print(f"\n=== {res.strategy} on {res.symbol} (daily) ===")
    print(f"Period:            {bars.index[0].date()} -> {bars.index[-1].date()}")
    print(f"Start -> End:      ${res.start_equity:,.0f} -> ${res.end_equity:,.0f}")
    print(f"Total return:      {res.total_return_pct:+.1f}%")
    print(f"CAGR:              {res.cagr_pct:+.1f}%")
    print(f"Sharpe:            {res.sharpe:.2f}")
    print(f"Max drawdown:      {res.max_drawdown_pct:.1f}%")
    print(f"Trades:            {res.num_trades}")
    print(f"Win rate:          {res.win_rate_pct:.1f}%")
    print(f"Profit factor:     {res.profit_factor:.2f}")

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    res.equity_curve.to_csv(out / "equity_vix_fear_SPY.csv")
    print("\nEquity curve saved to results/equity_vix_fear_SPY.csv")


if __name__ == "__main__":
    main()
