#!/usr/bin/env python3
"""Paper-trading runner. One iteration per invocation (cron-friendly), or --loop.

Usage:
    python scripts/run_paper.py --broker alpaca --symbols AAPL,SPY --strategy swing --timeframe 1d
    python scripts/run_paper.py --broker deribit --symbols BTC-PERPETUAL --strategy momentum --timeframe 1h --loop 3600
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.brokers.alpaca import AlpacaBroker
from bot.brokers.deribit import DeribitBroker
from bot.brokers.simulated import SimulatedBroker
from bot.config import Config
from bot.execution import ACExecutor
from bot.models import Order
from bot.risk import RiskManager
from bot.strategies.momentum import MomentumBreakout
from bot.strategies.ml_strategy import MLStrategy
from bot.strategies.options_momentum import OptionsMomentum
from bot.strategies.swing import SwingTrend

STRATEGIES = {
    "momentum": MomentumBreakout,
    "swing": SwingTrend,
    "options_momentum": OptionsMomentum,
    "ml": None,  # constructed separately; needs --model-path
}


def make_broker(name: str, cfg: Config):
    if name == "alpaca":
        return AlpacaBroker(cfg)
    if name == "deribit":
        return DeribitBroker(cfg)
    if name == "simulated":
        return SimulatedBroker(cfg)
    raise ValueError(f"unknown broker {name}")


def iterate(broker, strategy, symbols, risk, timeframe, executor=None, data=None):
    acct = broker.get_account()
    risk.new_day(datetime.now(timezone.utc).date(), acct.equity)
    positions = {p.symbol: p for p in broker.get_positions()}
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] equity=${acct.equity:,.0f} "
          f"positions={len(positions)} halted={risk.halted(acct.equity)}")

    for symbol in symbols:
        try:
            if data is not None and symbol in data:
                # simulated broker: bars come from the pre-fetched shim
                bars = data[symbol].tail(200)
                broker.set_price(symbol, float(bars["close"].iloc[-1]))
            else:
                bars = broker.get_bars(symbol, timeframe=timeframe, limit=200)
        except Exception as e:
            print(f"  {symbol}: data error: {e}")
            continue
        price = float(bars["close"].iloc[-1])

        # manage existing position stops -- always immediate, never scheduled
        pos = positions.get(symbol)
        if pos and risk.check_stops(symbol, price, pos.qty):
            side = "sell" if pos.qty > 0 else "buy"
            f = broker.place_order(Order(symbol=symbol, side=side, qty=abs(pos.qty),
                                         asset_class=pos.asset_class, reduce_only=True))
            print(f"  {symbol}: STOP/TRIGGER exit @ {f.price:.2f}")
            risk.clear_stops(symbol)
            continue

        for sig in strategy.on_bar(symbol, bars):
            # options signals need contract selection
            if sig.asset_class == "option":
                strat = strategy
                con = strat.pick_contract(sig.underlying or symbol, sig.action,
                                          broker.list_option_contracts,
                                          broker.get_option_snapshot)
                if not con:
                    print(f"  {symbol}: no suitable option contract found")
                    continue
                snap = broker.get_option_snapshot(con["symbol"])
                opt_px = float(snap.get("ask") or snap.get("mark") or 0)
                order = risk.size_order(
                    type(sig)(symbol=con["symbol"], action=sig.action, strength=sig.strength,
                              reason=sig.reason, asset_class="option", underlying=symbol),
                    opt_px, acct.equity, list(positions.values()))
            else:
                order = risk.size_order(sig, price, acct.equity, list(positions.values()))
            if order is None:
                print(f"  {symbol}: signal {sig.action} blocked by risk")
                continue
            # Execution routing: naive = legacy immediate market order (default,
            # behavior unchanged). twap/ac = schedule child slices via ACExecutor;
            # option orders always execute immediately (no AC calibration there).
            if executor is not None and order.asset_class != "option":
                exec_id = executor.submit(order, bars, arrival_price=price)
                print(f"  {symbol}: {order.side} {order.qty} scheduled "
                      f"({executor.mode}, id={exec_id}, {sig.reason})")
                continue
            f = broker.place_order(order)
            print(f"  {symbol}: {order.side} {order.qty} @ {f.price:.2f} ({sig.reason})")

    # fire any due execution slices (also picks up slices scheduled by earlier
    # cron invocations, since ACExecutor persists state to disk)
    if executor is not None:
        fills = executor.work(broker)
        for f in fills:
            print(f"  {f.symbol}: slice fill {f.side} {f.qty} @ {f.price:.2f}")
        if executor.pending():
            print(f"  execution: {executor.pending()} slices still working")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", choices=["alpaca", "deribit", "simulated"], default="alpaca")
    ap.add_argument("--symbols", default="AAPL,SPY")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="swing")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--loop", type=int, default=0, help="seconds between iterations; 0 = run once")
    ap.add_argument("--model-path", default="", help="path to trained .joblib (required for --strategy ml)")
    ap.add_argument("--execution", choices=["naive", "twap", "ac"], default="naive",
                    help="parent-order execution: naive=immediate market (default, legacy "
                         "behavior); twap/ac=schedule child slices via ACExecutor")
    ap.add_argument("--exec-horizon-s", type=float, default=600,
                    help="parent-order working horizon in seconds (twap/ac only)")
    ap.add_argument("--exec-slices", type=int, default=10,
                    help="number of child slices per parent (twap/ac only)")
    args = ap.parse_args()

    cfg = Config()
    cfg.assert_paper_safe()
    broker = make_broker(args.broker, cfg)
    if args.strategy == "ml":
        if not args.model_path:
            ap.error("--strategy ml requires --model-path models/ml_<SYMBOL>_1h.joblib")
        strategy = MLStrategy(args.model_path)
    else:
        strategy = STRATEGIES[args.strategy]()
    risk = RiskManager(cfg.starting_equity, cfg.risk_per_trade_pct, cfg.max_position_pct,
                       cfg.max_daily_loss_pct, cfg.max_open_positions,
                       cfg.stop_loss_pct, cfg.take_profit_pct)
    symbols = [s.strip() for s in args.symbols.split(",")]

    # simulated broker: pre-fetch bars once (no live market-data API on the sim)
    data = None
    if args.broker == "simulated":
        from bot.data.binance import fetch_klines
        data = {}
        for s in symbols:
            data[s] = fetch_klines(s, args.timeframe, total=400)
            print(f"  {s}: {len(data[s])} bars loaded "
                  f"({data[s].index.min()} -> {data[s].index.max()})")

    executor = None
    if args.execution in ("twap", "ac"):
        executor = ACExecutor(mode=args.execution, horizon_s=args.exec_horizon_s,
                              n_slices=args.exec_slices)
        print(f"Execution scheduling: mode={args.execution} "
              f"horizon={args.exec_horizon_s:.0f}s slices={args.exec_slices} "
              f"(fills logged to results/execution_fills.csv)")

    print(f"Paper trading: broker={args.broker} strategy={strategy.name} symbols={symbols}")
    while True:
        try:
            iterate(broker, strategy, symbols, risk, args.timeframe,
                    executor=executor, data=data)
        except Exception:
            traceback.print_exc()
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
