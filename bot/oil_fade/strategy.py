"""Oil spike-fade strategy core (Model D / D2).

Pure logic, no I/O: feed it 5-minute bars, it emits signals. The same code
drives the simulator and the live-paper runner, so sim results describe the
paper bot exactly.

Model D2 (frozen, preregistered 2026-09-28):
  - At the regular-session open, compute the overnight gap vs prior close.
  - Arm only if gap <= gap_max_pct (default 0.5). gap_max_pct=None disables
    the filter (plain Model D).
  - Short at the first 5m close >= trigger_pct above the open.
  - Stop-loss stop_pct above entry. Otherwise cover at exit_time_et.
  - One trade per day, flat overnight, short-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import List, Optional


@dataclass(frozen=True)
class OilFadeConfig:
    symbol: str = "USO"
    trigger_pct: float = 1.5      # short when a 5m close >= this % above open
    stop_pct: float = 0.75        # stop-loss distance above entry
    gap_max_pct: Optional[float] = 0.5  # arm only if overnight gap <= this; None = no filter
    exit_time_et: str = "15:55"   # cover shorts at/after this ET time
    risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0

    def label(self) -> str:
        g = f"gap{self.gap_max_pct}" if self.gap_max_pct is not None else "nogap"
        return (f"oilfade_trig{self.trigger_pct}_stop{self.stop_pct}_"
                f"{g}_exit{self.exit_time_et.replace(':', '')}")


@dataclass
class Bar:
    ts: datetime          # ET, bar open time
    open: float
    high: float
    low: float
    close: float


@dataclass
class TradeSignal:
    action: str           # 'short' | 'cover' | 'none'
    price: float          # reference price (bar close / stop level)
    reason: str           # 'trigger' | 'stop' | 'time_exit' | ...
    bar_ts: Optional[datetime] = None


class OilFadeStrategy:
    """Intraday state machine. One instance per trading day."""

    def __init__(self, cfg: OilFadeConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.session_open: Optional[float] = None
        self.armed = False
        self.gap_pct: Optional[float] = None
        self.position = 0            # 0 flat, -1 short
        self.done = False            # True after a round trip: no re-entry same day
        self.entry: Optional[float] = None
        self.stop: Optional[float] = None
        self._exit_t = time(*map(int, self.cfg.exit_time_et.split(":")))

    # ---- day lifecycle ----
    def new_day(self, session_open: float, prior_close: Optional[float]) -> None:
        self.reset()
        self.session_open = session_open
        filt = self.cfg.gap_max_pct
        if prior_close:
            self.gap_pct = (session_open / prior_close - 1) * 100
            self.armed = (filt is None) or (self.gap_pct <= filt)
        else:
            # Fail safe: gap is unmeasurable (prior close unavailable) -> keep the
            # day disarmed when the gap filter is enabled. Trading blind on such
            # days would trade outside the frozen D2 prereg, which only permits
            # skipping gap-up days, never assuming a measured flat gap.
            self.gap_pct = None
            self.armed = (filt is None)

    # ---- per-bar ----
    def on_bar(self, bar: Bar) -> TradeSignal:
        if self.session_open is None:
            raise RuntimeError("new_day() must be called before on_bar()")
        if self.position == 0:
            if self.done or not self.armed:
                return TradeSignal("none", bar.close, "waiting", bar.ts)
            if bar.close >= self.session_open * (1 + self.cfg.trigger_pct / 100):
                self.position = -1
                self.entry = bar.close
                self.stop = self.entry * (1 + self.cfg.stop_pct / 100)
                return TradeSignal("short", bar.close, "trigger", bar.ts)
            return TradeSignal("none", bar.close, "waiting", bar.ts)
        # short: stop check first (conservative: any touch fills)
        if self.stop is not None and bar.high >= self.stop:
            self.position = 0
            self.done = True
            return TradeSignal("cover", self.stop, "stop", bar.ts)
        if bar.ts.time() >= self._exit_t:
            self.position = 0
            self.done = True
            return TradeSignal("cover", bar.close, "time_exit", bar.ts)
        return TradeSignal("none", bar.close, "holding", bar.ts)

    @property
    def is_flat(self) -> bool:
        return self.position == 0
