"""Broker interface. Every venue (paper, testnet, backtest-sim) implements this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import Account, Fill, Order, Position


class Broker(ABC):
    name: str = "base"

    # ---- account / positions ----
    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> List[Position]: ...

    # ---- market data ----
    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        """Return OHLCV bars, datetime index ascending, columns open/high/low/close/volume."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> float: ...

    # ---- orders ----
    @abstractmethod
    def place_order(self, order: Order) -> Fill: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    # ---- options support (optional; raise NotImplementedError if unsupported) ----
    def list_option_contracts(self, underlying: str, **filters: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError(f"{self.name} does not support option chains")

    def get_option_snapshot(self, option_symbol: str) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not support option snapshots")
