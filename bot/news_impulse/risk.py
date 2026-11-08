"""Event-level risk: per-event sizing, concurrency cap, daily kill switch.

Separate from bot/risk.py (which guards portfolio-level trade risk): this
module answers "may we open a NEW event position right now?" Paper-only;
the runner, not this module, enforces halts by refusing to emit signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Set


@dataclass
class EventRiskManager:
    equity: float
    risk_per_event_pct: float = 0.5
    max_concurrent: int = 3
    max_daily_loss_pct: float = 2.0

    _day: date = field(default=None)  # type: ignore
    _day_start: float = 0.0
    _day_pnl: float = 0.0
    _open: Set[str] = field(default_factory=set)
    halted: bool = False

    def new_day(self, today: date, equity: float) -> None:
        if self._day != today:
            self._day = today
            self._day_start = equity
            self._day_pnl = 0.0
            self.halted = False

    def can_open(self, symbol: str, today: date, equity: float) -> bool:
        self.new_day(today, equity)
        if self.halted:
            return False
        if symbol in self._open:
            return False
        if len(self._open) >= self.max_concurrent:
            return False
        return True

    def register_open(self, symbol: str) -> None:
        self._open.add(symbol)

    def register_close(self, symbol: str, pnl: float, today: date, equity: float) -> None:
        self.new_day(today, equity)
        self._open.discard(symbol)
        self._day_pnl += pnl
        if -self._day_pnl >= self.max_daily_loss_pct / 100.0 * self._day_start:
            self.halted = True

    def size_qty(self, price: float, stop_distance: float) -> float:
        """Shares/contracts so that stop_distance loss == risk_per_event_pct."""
        risk_dollars = self.equity * self.risk_per_event_pct / 100.0
        if price <= 0 or stop_distance <= 0:
            return 0.0
        return risk_dollars / stop_distance
