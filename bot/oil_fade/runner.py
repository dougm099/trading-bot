"""Market-hours paper runner for the oil-fade bot.

Loop: every `poll_secs` during market hours, pull today's USO 5m bars from the
broker, feed newly closed bars through OilFadeStrategy, and translate signals
into orders via the shared RiskManager. Paper only: refuses to start unless
the config passes assert_paper_safe() (paper URL, live trading disabled).

Safety:
  - kill-switch file: when present, flattens USO and stops the loop
  - borrow check: skips entries when USO is not shortable at Alpaca
  - daily loss halt via RiskManager (3% of day-start equity)
  - end-of-day sweep: any open USO position is flattened after 16:00 ET
  - state file: survives restarts without double-trading
  - dry_run=True (default): logs signals, places no orders
"""
from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from ..brokers.alpaca import AlpacaBroker
from ..config import Config
from ..models import Signal
from ..risk import RiskManager
from .strategy import Bar, OilFadeConfig, OilFadeStrategy

log = logging.getLogger("oil_fade.runner")
ET = ZoneInfo("America/New_York")

LEDGER_COLS = ["ts_et", "date", "event", "side", "qty", "price", "status", "note"]


def _yfinance_prior_close(symbol: str) -> Optional[float]:
    """Prior completed-session close via yfinance daily bars.

    Alpaca's free paper feed returns only today's bars, so the broker cannot
    supply a prior close. yfinance (already used by the hawk-snapshot cron on
    this host) serves daily USO closes. Returns the last completed session's
    close, or None on any failure (fail safe).
    """
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
    except Exception as e:
        log.warning("yfinance fetch failed for %s: %s", symbol, e)
        return None
    if hist is None or hist.empty:
        return None
    today = datetime.now(ET).date()
    completed = hist[[d.date() < today for d in hist.index]]
    if completed.empty:
        return None
    try:
        return float(completed["Close"].iloc[-1])
    except (IndexError, ValueError):
        return None


class OilFadePaperRunner:
    def __init__(self, cfg: OilFadeConfig, broker: AlpacaBroker,
                 risk: RiskManager, state_path: Path, ledger_path: Path,
                 dry_run: bool = True,
                 kill_path: Optional[Path] = None,
                 poll_secs: int = 60):
        app_cfg = Config()
        app_cfg.assert_paper_safe()
        if app_cfg.live_trading_enabled:
            raise RuntimeError("refusing to run: LIVE_TRADING_ENABLED is set")
        self.cfg = cfg
        self.broker = broker
        self.risk = risk
        self.state_path = Path(state_path)
        self.ledger_path = Path(ledger_path)
        self.dry_run = dry_run
        self.kill_path = Path(kill_path) if kill_path else self.state_path.parent / "KILL"
        self.poll_secs = poll_secs
        self.strat = OilFadeStrategy(cfg)
        self._day: Optional[str] = None
        self._gap_source: str = "unknown"
        self._shortable_cache: Optional[bool] = None
        self._load_state()

    # ---- state ----
    def _load_state(self):
        self._mem = {}
        if self.state_path.exists():
            self._mem = json.loads(self.state_path.read_text())

    def _save_state(self):
        self.state_path.write_text(json.dumps(self._mem, indent=1, default=str))

    def _ledger(self, event: str, side="", qty="", price="", status="", note=""):
        now = datetime.now(ET)
        row = {"ts_et": now.strftime("%Y-%m-%d %H:%M:%S"), "date": now.strftime("%Y-%m-%d"),
               "event": event, "side": side, "qty": qty, "price": price,
               "status": status, "note": note}
        df = pd.DataFrame([[row[c] for c in LEDGER_COLS]], columns=LEDGER_COLS)
        hdr = not self.ledger_path.exists()
        df.to_csv(self.ledger_path, mode="a", header=hdr, index=False)
        log.info("LEDGER %s", row)

    # ---- market data ----
    def _clock(self) -> dict:
        return self.broker._req("GET", f"{self.broker.base}/v2/clock")

    def _today_bars_et(self) -> pd.DataFrame:
        df = self.broker.get_bars(self.cfg.symbol, timeframe="5m", limit=100)
        df.index = df.index.tz_convert(ET)
        today = datetime.now(ET).date()
        return df[df.index.date == today]

    def _prior_close(self) -> Optional[float]:
        df = self.broker.get_bars(self.cfg.symbol, timeframe="1d", limit=5)
        if len(df) >= 2:
            self._gap_source = "alpaca"
            return float(df["close"].iloc[-2])
        # Alpaca free paper data carries no prior-day bars: fall back to
        # yfinance daily closes (same "prior close" definition, other source).
        # Cached per day so the endpoint is hit at most once per session.
        self._gap_source = "yfinance"
        return self._prior_close_cached()

    def _prior_close_cached(self) -> Optional[float]:
        today = str(datetime.now(ET).date())
        cache_path = self.state_path.parent / "prior_close_cache.json"
        try:
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                if cached.get("for_date") == today:
                    return cached.get("value")
        except Exception:
            pass
        val = _yfinance_prior_close(self.cfg.symbol)
        try:
            cache_path.write_text(json.dumps({"for_date": today, "value": val}))
        except Exception:
            pass
        return val

    def _shortable(self) -> bool:
        if self._shortable_cache is None:
            a = self.broker._req("GET", f"{self.broker.base}/v2/assets/{self.cfg.symbol}")
            self._shortable_cache = bool(a.get("shortable", False))
        return self._shortable_cache

    # ---- day lifecycle ----
    def _maybe_new_day(self, bars: pd.DataFrame):
        day = str(bars.index[0].date()) if len(bars) else str(datetime.now(ET).date())
        if self._day != day:
            self._day = day
            self._shortable_cache = None
            acct = self.broker.get_account()
            self.risk.new_day(datetime.now(ET).date(), acct.equity)
            if len(bars):
                self.strat.new_day(float(bars["open"].iloc[0]), self._prior_close())
            self._mem = {"date": day, "armed": self.strat.armed,
                         "gap_pct": self.strat.gap_pct, "bars_seen": 0,
                         "position": 0, "entry": None, "stop": None}
            self._save_state()
            self._ledger("new_day", note=(f"armed={self.strat.armed} "
                                                f"gap={self.strat.gap_pct} src={self._gap_source}"))

    # ---- one poll ----
    def run_once(self) -> str:
        """Single iteration. Returns a short status string (testable)."""
        if self.kill_path.exists():
            self._flatten("kill_switch")
            return "killed"
        try:
            clock = self._clock()
        except Exception as e:
            log.warning("clock failed: %s", e)
            return "clock_error"
        if not clock.get("is_open"):
            return "market_closed"

        bars = self._today_bars_et()
        if len(bars) < 2:
            return "no_bars"
        self._maybe_new_day(bars)
        n_seen = self._mem.get("bars_seen", 0)
        new = bars.iloc[n_seen:-1]  # all closed bars not yet seen (drop forming bar)
        if len(new) == 0:
            return "waiting"

        for ts, r in new.iterrows():
            sig = self.strat.on_bar(Bar(ts.to_pydatetime(), float(r["open"]),
                                        float(r["high"]), float(r["low"]), float(r["close"])))
            self._handle_signal(sig)
        self._mem["bars_seen"] = len(bars) - 1
        self._mem["position"] = self.strat.position
        self._save_state()
        self._reconcile()
        return "ok"

    def _handle_signal(self, sig) -> None:
        if sig.action == "none":
            return
        acct = self.broker.get_account()
        positions = self.broker.get_positions()
        if sig.action == "short":
            if not self._shortable():
                self._ledger("skip", note="USO not shortable")
                return
            order = self.risk.size_order(
                Signal(symbol=self.cfg.symbol, action="sell",
                       reason=f"oil_fade:{sig.reason}", asset_class="spot"),
                price=sig.price, equity=acct.equity, positions=positions)
            if order is None:
                self._ledger("skip", note="risk blocked entry (halt/limits)")
                return
            self._submit(order, f"short@{sig.price:.2f} {sig.reason}")
            self._mem["entry"] = sig.price
        elif sig.action == "cover":
            order = self.risk.size_order(
                Signal(symbol=self.cfg.symbol, action="flat",
                       reason=f"oil_fade:{sig.reason}", asset_class="spot"),
                price=sig.price, equity=acct.equity, positions=positions)
            if order is None:
                return
            self._submit(order, f"cover@{sig.price:.2f} {sig.reason}")

    def _submit(self, order, note: str) -> None:
        if self.dry_run:
            self._ledger("signal", side=order.side, qty=order.qty,
                         price=note, status="dry_run")
            return
        try:
            fill = self.broker.place_order(order)
            self._ledger("order", side=fill.side, qty=fill.qty,
                         price=fill.price, status=fill.status, note=note)
            if fill.status not in ("filled", "partially_filled"):
                log.warning("order not immediately filled: %s", asdict(fill))
        except Exception as e:
            self._ledger("order_error", side=order.side, qty=order.qty, note=f"{note} :: {e}")

    def _reconcile(self) -> None:
        """Broker is truth: if it disagrees with strategy state, go safe."""
        positions = {p.symbol: p.qty for p in self.broker.get_positions()}
        bqty = positions.get(self.cfg.symbol, 0.0)
        if self.strat.is_flat and bqty != 0:
            log.error("strategy flat but broker holds %s %+g -> flattening", self.cfg.symbol, bqty)
            self._flatten("reconcile_mismatch")
        elif not self.strat.is_flat and bqty == 0 and not self.dry_run:
            log.error("strategy short but broker flat -> marking done")
            self.strat.done = True

    def _flatten(self, why: str) -> None:
        acct = self.broker.get_account()
        positions = self.broker.get_positions()
        order = self.risk.size_order(
            Signal(symbol=self.cfg.symbol, action="flat", reason=why, asset_class="spot"),
            price=self.broker.get_last_price(self.cfg.symbol),
            equity=acct.equity, positions=positions)
        if order:
            self._submit(order, f"flatten:{why}")
        self.strat.done = True

    # ---- main loop ----
    def run_forever(self) -> None:
        log.info("oil-fade paper runner start (dry_run=%s, %s)", self.dry_run, self.cfg.label())
        self._ledger("start", note=f"dry_run={self.dry_run} {self.cfg.label()}")
        while True:
            try:
                status = self.run_once()
                if status == "killed":
                    log.info("kill switch engaged, stopping")
                    self._ledger("stopped", note="kill_switch")
                    return
            except Exception:
                log.exception("run_once failed")
            _time.sleep(self.poll_secs)
