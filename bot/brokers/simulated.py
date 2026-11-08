"""Simulated broker for backtests: fills at next-bar open with slippage."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd

from ..config import Config
from ..models import Account, Fill, Order, Position
from .base import Broker


class SimulatedBroker(Broker):
    name = "simulated"

    def __init__(self, cfg: Config, allow_short: bool = True):
        self.cfg = cfg
        self.cash = cfg.starting_equity
        self.allow_short = allow_short
        self.positions: Dict[str, Position] = {}
        self.prices: Dict[str, float] = {}
        self.fills: List[Fill] = []
        self.realized_pnls: List[float] = []  # per-close realized P&L
        self._pending: List[Order] = []

    # ---- test harness hooks ----
    def set_price(self, symbol: str, price: float) -> None:
        self.prices[symbol] = price

    def mark_to_market(self, symbol: str, price: float) -> None:
        self.prices[symbol] = price
        p = self.positions.get(symbol)
        if p:
            p.current_price = price
            p.unrealized_pnl = (price - p.avg_entry) * p.qty

    # ---- account / positions ----
    def get_account(self) -> Account:
        # equity = settled cash + signed market value of positions
        # (shorts are a liability: qty < 0 subtracts automatically)
        equity = self.cash + sum(p.qty * p.current_price for p in self.positions.values())
        return Account(equity=equity, cash=self.cash, buying_power=self.cash)

    def get_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.qty != 0]

    # ---- market data (not used in backtest; engine feeds bars directly) ----
    def get_bars(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        raise NotImplementedError("SimulatedBroker does not fetch bars; use data loaders")

    def get_last_price(self, symbol: str) -> float:
        return self.prices[symbol]

    # ---- orders: fill immediately at current price +/- slippage ----
    def place_order(self, order: Order) -> Fill:
        px = self.prices[order.symbol]
        slip = px * (self.cfg.slippage_bps / 10_000)
        fill_px = (order.limit_price or (px + slip if order.side == "buy" else px - slip))
        commission = self.cfg.commission_per_trade

        pos = self.positions.get(order.symbol)
        qty = pos.qty if pos else 0.0
        avg = pos.avg_entry if pos else 0.0

        if order.side == "buy":
            if qty < 0:
                # 1) cover existing short: pay to buy back
                cover = min(order.qty, -qty)
                self.cash -= fill_px * cover
                self.realized_pnls.append((avg - fill_px) * cover)
                qty += cover
                if qty == 0:
                    avg = 0.0
                remaining = order.qty - cover
            else:
                remaining = order.qty
            # 2) open/add long with the remainder
            if remaining > 0:
                if self.cash < fill_px * remaining:
                    raise RuntimeError("insufficient cash")
                self.cash -= fill_px * remaining
                avg = (avg * qty + fill_px * remaining) / (qty + remaining) if qty + remaining else 0.0
                qty += remaining
        else:  # sell
            if qty > 0:
                # 1) close existing long: receive proceeds
                close = min(order.qty, qty)
                self.cash += fill_px * close
                self.realized_pnls.append((fill_px - avg) * close)
                qty -= close
                if qty == 0:
                    avg = 0.0
                remaining = order.qty - close
            else:
                remaining = order.qty
            # 2) open/add short with the remainder
            if remaining > 0:
                if not self.allow_short:
                    raise RuntimeError("short selling disabled in this simulation")
                self.cash += fill_px * remaining  # short proceeds held as cash
                avg = (avg * -qty + fill_px * remaining) / (-qty + remaining) if (-qty + remaining) else 0.0
                qty -= remaining

        if qty == 0:
            self.positions.pop(order.symbol, None)
        else:
            self.positions[order.symbol] = Position(order.symbol, qty, avg, fill_px, 0.0, order.asset_class)
        self.cash -= commission
        fill = Fill(order_id=uuid4().hex[:12], symbol=order.symbol, side=order.side,
                    qty=order.qty, price=fill_px, timestamp=datetime.now(timezone.utc),
                    commission=commission)
        self.fills.append(fill)
        return fill

    def cancel_order(self, order_id: str) -> None:
        pass
