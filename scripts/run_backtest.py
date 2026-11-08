#!/usr/bin/env python3
"""Run a backtest on real market data and print the report.

Usage:
    python scripts/run_backtest.py --symbol BTCUSDT --strategy momentum --bars 3000
    python scripts/run_backtest.py --symbol ETHUSDT --strategy swing --bars 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.backtest import run_backtest
from bot.config import Config
from bot.data.binance import fetch_klines
from bot.strategies.momentum import MomentumBreakout
from bot.strategies.swing import SwingTrend

STRATEGIES = {"momentum": MomentumBreakout, "swing": SwingTrend}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="momentum")
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--no-short", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    print(f"Fetching {args.bars} hourly bars for {args.symbol} ...")
    bars = fetch_klines(args.symbol, "1h", total=args.bars)
    print(f"Got {len(bars)} bars: {bars.index[0]} -> {bars.index[-1]}")

    strat = STRATEGIES[args.strategy]()
    res = run_backtest(strat, args.symbol, bars, cfg, allow_short=not args.no_short)

    print(f"\n=== {res.strategy} on {res.symbol} ===")
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
    res.equity_curve.to_csv(out / f"equity_{args.strategy}_{args.symbol}.csv")
    print(f"\nEquity curve saved to results/equity_{args.strategy}_{args.symbol}.csv")


if __name__ == "__main__":
    main()
