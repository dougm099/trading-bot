"""Impulse confirmation engine: trade the market's reaction, not the text.

After a verified event, record a reference price and watch price action for up
to max_wait_min. Enter only if the move exceeds impulse_mult × the recent
impulse_lookback_min realized volatility, in a consistent direction. Exits:
hard time stop (hold_min), trailing stop at 1× the entry impulse, hard stop at
0.5× the entry impulse against the position.

Emits bot.models.Signal objects (buy/sell/flat). Pure logic: price ticks are
fed in by the runner; this module never touches a broker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..models import Signal
from .events import NewsEvent


@dataclass
class Watch:
    symbol: str
    event: NewsEvent
    detected_at: datetime
    ref_price: float
    ref_vol: float          # realized vol per minute over lookback window
    state: str = "watching"  # watching | long | short | done
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    peak: float = 0.0       # favorable excursion peak for trailing stop
    impulse: float = 0.0    # signed entry impulse (price units)


class ImpulseEngine:
    def __init__(self, impulse_mult: float = 2.0, max_wait_min: int = 10,
                 hold_min: int = 10):
        self.impulse_mult = impulse_mult
        self.max_wait = timedelta(minutes=max_wait_min)
        self.hold = timedelta(minutes=hold_min)
        self.watches: Dict[str, Watch] = {}
        # per-symbol minute close history for vol estimation (fed by runner)
        self._closes: Dict[str, List[float]] = {}

    # ---- price feed -----------------------------------------------------
    def on_minute_close(self, symbol: str, close: float) -> None:
        hist = self._closes.setdefault(symbol, [])
        hist.append(close)
        if len(hist) > 60:
            del hist[:-60]

    def realized_vol_per_min(self, symbol: str, lookback: int = 5) -> float:
        hist = self._closes.get(symbol, [])
        if len(hist) < lookback + 1:
            return 0.0
        rets = [math.log(hist[i] / hist[i - 1]) for i in range(-lookback, 0)]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    # ---- event handling -------------------------------------------------
    def register_event(self, event: NewsEvent, ref_price: float) -> None:
        if event.symbol in self.watches and self.watches[event.symbol].state != "done":
            return  # one live watch per symbol
        vol = self.realized_vol_per_min(event.symbol)
        self.watches[event.symbol] = Watch(
            symbol=event.symbol, event=event, detected_at=event.first_seen,
            ref_price=ref_price, ref_vol=vol,
        )

    def on_tick(self, symbol: str, price: float, ts: datetime) -> List[Signal]:
        """Feed each price tick; returns signals (possibly empty)."""
        w = self.watches.get(symbol)
        if w is None or w.state == "done":
            return []
        out: List[Signal] = []

        if w.state == "watching":
            if ts - w.detected_at > self.max_wait:
                w.state = "done"
                return out
            move = (price - w.ref_price) / w.ref_price if w.ref_price else 0.0
            thresh = self.impulse_mult * w.ref_vol
            if w.ref_vol > 0 and abs(move) >= thresh:
                direction = "buy" if move > 0 else "sell"
                w.state = "long" if move > 0 else "short"
                w.entry_price = price
                w.entry_time = ts
                w.peak = price
                w.impulse = price - w.ref_price
                out.append(Signal(symbol=symbol, action=direction, strength=1.0,
                                  reason=f"news-impulse {w.event.category} "
                                         f"move={move:.4f} thr={thresh:.4f}"))
            return out

        # in position: manage exits
        assert w.entry_time is not None
        if w.state == "long":
            w.peak = max(w.peak, price)
        else:
            w.peak = min(w.peak, price)

        stop_dist = 0.5 * abs(w.impulse)
        trail_dist = 1.0 * abs(w.impulse)
        exit_reason = ""
        if ts - w.entry_time >= self.hold:
            exit_reason = "time-stop"
        elif w.state == "long" and (price <= w.entry_price - stop_dist
                                    or price <= w.peak - trail_dist):
            exit_reason = "stop" if price <= w.entry_price - stop_dist else "trailing-stop"
        elif w.state == "short" and (price >= w.entry_price + stop_dist
                                     or price >= w.peak + trail_dist):
            exit_reason = "stop" if price >= w.entry_price + stop_dist else "trailing-stop"

        if exit_reason:
            out.append(Signal(symbol=symbol, action="flat", strength=1.0,
                              reason=f"news-impulse exit {exit_reason}"))
            w.state = "done"
        return out
