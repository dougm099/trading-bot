"""Risk manager: position sizing + portfolio guards.

- Sizes each trade from risk_per_trade_pct of equity and the strategy's stop distance
- Caps single-position size (max_position_pct)
- Halts new entries after max_daily_loss_pct drawdown from day-start equity
- Caps number of concurrent positions
- Attaches stop-loss / take-profit levels to every position
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from .models import Order, Position, Signal


@dataclass
class StopLevels:
    stop: float
    take_profit: float


class RiskManager:
    def __init__(self, starting_equity: float, risk_per_trade_pct: float = 1.0,
                 max_position_pct: float = 20.0, max_daily_loss_pct: float = 3.0,
                 max_open_positions: int = 5, stop_loss_pct: float = 2.0,
                 take_profit_pct: float = 6.0):
        self.starting_equity = starting_equity
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self._day: Optional[date] = None
        self._day_start_equity = starting_equity
        self.stops: Dict[str, StopLevels] = {}

    def new_day(self, today: date, equity: float) -> None:
        if self._day != today:
            self._day = today
            self._day_start_equity = equity

    def halted(self, equity: float) -> bool:
        dd = (self._day_start_equity - equity) / self._day_start_equity * 100
        return dd >= self.max_daily_loss_pct

    def size_order(self, signal: Signal, price: float, equity: float,
                   positions: List[Position]) -> Optional[Order]:
        """Turn a signal into a sized order, or None if a guard blocks it."""
        existing = next((p for p in positions if p.symbol == signal.symbol), None)
        ex_qty = existing.qty if existing else 0.0

        if signal.action == "flat":
            if not existing or ex_qty == 0:
                return None
            return Order(symbol=signal.symbol, side="sell" if ex_qty > 0 else "buy",
                         qty=abs(ex_qty), asset_class=existing.asset_class, reduce_only=True)

        # opposing signal on an open position -> close it (never flip in one order)
        if existing and ex_qty != 0:
            if (signal.action == "buy" and ex_qty < 0) or (signal.action == "sell" and ex_qty > 0):
                return Order(symbol=signal.symbol, side="sell" if ex_qty > 0 else "buy",
                             qty=abs(ex_qty), asset_class=existing.asset_class, reduce_only=True)

        if self.halted(equity):
            return None
        if len([p for p in positions if p.qty != 0]) >= self.max_open_positions \
                and ex_qty == 0:
            return None

        risk_dollars = equity * (self.risk_per_trade_pct / 100) * signal.strength
        stop_dist = price * (self.stop_loss_pct / 100)
        qty = risk_dollars / stop_dist if stop_dist > 0 else 0
        # cap TOTAL exposure (existing + new) at max_position_pct
        max_total = (equity * self.max_position_pct / 100) / price
        qty = min(qty, max(0.0, max_total - abs(ex_qty)))
        if qty * price < 10:  # dust filter: $10 minimum notional
            return None

        side = "buy" if signal.action == "buy" else "sell"
        if signal.asset_class == "option":
            qty = max(1, round(qty))  # whole contracts
        order = Order(symbol=signal.symbol, side=side, qty=round(qty, 6),
                      asset_class=signal.asset_class)
        if side == "buy":
            self.stops[signal.symbol] = StopLevels(
                stop=price * (1 - self.stop_loss_pct / 100),
                take_profit=price * (1 + self.take_profit_pct / 100))
        else:
            self.stops[signal.symbol] = StopLevels(
                stop=price * (1 + self.stop_loss_pct / 100),
                take_profit=price * (1 - self.take_profit_pct / 100))
        return order

    def check_stops(self, symbol: str, price: float, qty: float) -> Optional[str]:
        """Returns 'stop' | 'take_profit' | None for an open position."""
        lv = self.stops.get(symbol)
        if not lv or qty == 0:
            return None
        if qty > 0:
            if price <= lv.stop:
                return "stop"
            if price >= lv.take_profit:
                return "take_profit"
        else:
            if price >= lv.stop:
                return "stop"
            if price <= lv.take_profit:
                return "take_profit"
        return None

    def clear_stops(self, symbol: str) -> None:
        self.stops.pop(symbol, None)
