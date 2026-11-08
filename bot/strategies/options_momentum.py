"""Long-options momentum (equity options via Alpaca, paper-tradable).

On a momentum buy signal for the underlying, buy the nearest-expiry (7-30
DTE) ~0.50-delta call; on a sell signal, buy the put. Exits: +50% take
profit, -50% stop, DTE < 2, or opposite signal. Defined risk: max loss is the
premium paid.

Paper-trading only: options have no historical chain data for backtesting, so
this strategy is excluded from the backtest engine.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..models import Signal
from .base import Strategy
from .momentum import MomentumBreakout


class OptionsMomentum(Strategy):
    name = "options_momentum"
    backtestable = False

    def __init__(self, min_dte: int = 7, max_dte: int = 30,
                 target_delta: float = 0.50, tp_pct: float = 50.0, sl_pct: float = 50.0):
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.target_delta = target_delta
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self._mom = MomentumBreakout()

    def pick_contract(self, underlying: str, direction: str,
                      chain_fn: Callable[..., List[Dict[str, Any]]],
                      snap_fn: Callable[[str], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Choose the ATM contract in the DTE window closest to target delta."""
        today = date.today()
        kind = "call" if direction == "buy" else "put"
        contracts = chain_fn(underlying, option_type=kind)
        best, best_score = None, 1e9
        for con in contracts:
            exp = con.get("expiration_date")
            if not exp:
                continue
            dte = (datetime.fromisoformat(exp).date() - today).days
            if not (self.min_dte <= dte <= self.max_dte):
                continue
            snap = snap_fn(con["symbol"])
            greeks = (snap.get("greeks") or {})
            delta = abs(float(greeks.get("delta") or 0))
            if delta == 0:
                continue
            score = abs(delta - self.target_delta)
            if score < best_score:
                best, best_score = con, score
        return best

    def on_bar(self, symbol: str, bars: pd.DataFrame) -> List[Signal]:
        # underlying-level signal; the runner translates it into an option order
        sigs = self._mom.on_bar(symbol, bars)
        out = []
        for s in sigs:
            if s.action in ("buy", "sell"):
                out.append(Signal(symbol=s.symbol, action=s.action, strength=s.strength,
                                  reason="options_momentum: " + s.reason,
                                  asset_class="option", underlying=symbol))
            else:
                out.append(Signal(symbol=s.symbol, action="flat", reason=s.reason,
                                  asset_class="option", underlying=symbol))
        return out
