"""Core data models: signals, orders, positions, bars."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Signal:
    """A strategy's desired action on a symbol."""
    symbol: str                 # broker-native symbol (e.g. 'BTCUSDT', 'AAPL', OCC option symbol)
    action: str                 # 'buy' | 'sell' | 'flat'  (flat = close any position)
    strength: float = 1.0       # 0..1 conviction, used by risk manager for sizing
    reason: str = ""
    asset_class: str = "spot"   # 'spot' | 'option' | 'future' | 'perp'
    underlying: Optional[str] = None  # for derivatives: underlying symbol


@dataclass
class Order:
    symbol: str
    side: str                   # 'buy' | 'sell'
    qty: float
    order_type: str = "market"  # 'market' | 'limit'
    limit_price: Optional[float] = None
    time_in_force: str = "day"  # 'day' | 'gtc' | 'ioc'
    asset_class: str = "spot"
    reduce_only: bool = False
    client_id: Optional[str] = None


@dataclass
class Position:
    symbol: str
    qty: float                  # signed: + long, - short
    avg_entry: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    asset_class: str = "spot"


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    status: str = ""  # broker order status at submission (e.g. filled, accepted).
    # 'accepted' means the venue took the order but has not filled it yet
    # (e.g. market closed). Never mark it 'filled' until it actually is.
