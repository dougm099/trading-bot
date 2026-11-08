"""Backtest engine: walk bars forward, fill at next-bar open, track equity.

Fills use the next bar's open (no lookahead) with slippage from config.
Reports total return, CAGR, Sharpe, max drawdown, win rate, profit factor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .brokers.simulated import SimulatedBroker
from .config import Config
from .models import Order
from .risk import RiskManager
from .strategies.base import Strategy


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    start_equity: float
    end_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_drawdown_pct: float
    num_trades: int
    win_rate_pct: float
    profit_factor: float
    equity_curve: pd.Series = field(repr=False)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min() * 100)


def run_backtest(strategy: Strategy, symbol: str, bars: pd.DataFrame,
                 cfg: Config, allow_short: bool = True) -> BacktestResult:
    broker = SimulatedBroker(cfg, allow_short=allow_short)
    risk = RiskManager(cfg.starting_equity, cfg.risk_per_trade_pct,
                       cfg.max_position_pct, cfg.max_daily_loss_pct,
                       cfg.max_open_positions, cfg.stop_loss_pct,
                       cfg.take_profit_pct)
    equity_curve: List[float] = []

    for i in range(len(bars)):
        ts = bars.index[i]
        bar = bars.iloc[i]
        broker.mark_to_market(symbol, float(bar["close"]))
        # risk uses yesterday's close as today's tradable price via next-open fills
        risk.new_day(ts.date(), broker.get_account().equity)

        # 1) manage open positions: stops / take-profits at this bar
        for pos in list(broker.get_positions()):
            trig = risk.check_stops(pos.symbol, float(bar["close"]), pos.qty)
            if trig:
                broker.set_price(pos.symbol, float(bar["close"]))
                broker.place_order(Order(symbol=pos.symbol,
                                         side="sell" if pos.qty > 0 else "buy",
                                         qty=abs(pos.qty), asset_class=pos.asset_class,
                                         reduce_only=True))
                risk.clear_stops(pos.symbol)

        # 2) new signals, filled at NEXT bar open (i+1) to avoid lookahead
        if i + 1 < len(bars):
            next_open = float(bars["open"].iloc[i + 1])
            for sig in strategy.on_bar(symbol, bars.iloc[: i + 1]):
                if sig.asset_class != "spot":
                    continue  # derivatives need chain data; paper path only
                equity = broker.get_account().equity
                order = risk.size_order(sig, next_open, equity, broker.get_positions())
                if order is None:
                    continue
                broker.set_price(order.symbol, next_open)
                try:
                    broker.place_order(order)
                except RuntimeError:
                    continue
                if order.reduce_only:
                    risk.clear_stops(order.symbol)

        equity_curve.append(broker.get_account().equity)

    realized = broker.realized_pnls

    eq = pd.Series(equity_curve, index=bars.index)
    rets = eq.pct_change().dropna()
    years = max((bars.index[-1] - bars.index[0]).total_seconds() / 31_536_000, 1e-9)
    end = float(eq.iloc[-1])
    cagr = float((end / cfg.starting_equity) ** (1 / years) - 1) * 100 if end > 0 else -100.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(24 * 365)) if rets.std() > 0 else 0.0
    wins = [r for r in realized if r > 0]
    losses = [r for r in realized if r < 0]
    return BacktestResult(
        strategy=strategy.name, symbol=symbol,
        start_equity=cfg.starting_equity, end_equity=float(eq.iloc[-1]),
        total_return_pct=float((eq.iloc[-1] / cfg.starting_equity - 1) * 100),
        cagr_pct=cagr,
        sharpe=sharpe, max_drawdown_pct=_max_drawdown(eq),
        num_trades=len(realized),
        win_rate_pct=float(len(wins) / len(realized) * 100) if realized else 0.0,
        profit_factor=float(sum(wins) / abs(sum(losses))) if losses else float("inf"),
        equity_curve=eq,
    )
