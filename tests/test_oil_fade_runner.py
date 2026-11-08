"""Runner tests with a scripted fake broker (no network)."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from bot.config import Config
from bot.models import Account, Fill, Order, Position
from bot.oil_fade.runner import OilFadePaperRunner
from bot.oil_fade.strategy import OilFadeConfig
from bot.risk import RiskManager

ET = ZoneInfo("America/New_York")


class FakeBroker:
    name = "fake"
    base = "https://paper-api.alpaca.markets"

    def __init__(self, bars_5m, one_daily_bar=False):
        self._bars = bars_5m
        self.orders = []
        self.pos_qty = 0.0
        self._one_daily_bar = one_daily_bar

    def _req(self, method, url, **kw):
        if url.endswith("/v2/clock"):
            return {"is_open": True}
        if "/v2/assets/" in url:
            return {"shortable": True}
        raise AssertionError(url)

    def get_bars(self, symbol, timeframe="1h", limit=200):
        if timeframe == "5m":
            return self._bars
        # two-row stub: iloc[-2].close == 100 -> prior close 100 (gap 0, armed)
        if self._one_daily_bar:
            return pd.DataFrame(
                [{"open": 99, "high": 99, "low": 99, "close": 100, "volume": 1}],
                index=[datetime.now(ET).replace(hour=9, minute=0)])
        d = pd.DataFrame([{"open": 99, "high": 99, "low": 99, "close": 100, "volume": 1},
                          {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1}],
                         index=[datetime.now(ET).replace(hour=9, minute=0),
                                datetime.now(ET).replace(hour=9, minute=1)])
        return d

    def get_account(self):
        return Account(equity=100000.0, cash=100000.0, buying_power=200000.0)

    def get_positions(self):
        if self.pos_qty == 0:
            return []
        return [Position(symbol="USO", qty=self.pos_qty, avg_entry=101.6,
                         current_price=101.0, asset_class="spot")]

    def get_last_price(self, symbol):
        return 101.0

    def place_order(self, order: Order):
        self.orders.append(order)
        if order.side == "sell" and not order.reduce_only:
            self.pos_qty -= order.qty
        else:
            self.pos_qty = 0.0
        return Fill(order_id="f1", symbol=order.symbol, side=order.side,
                    qty=order.qty, price=101.0,
                    timestamp=datetime.now(ET), status="filled")


def make_bars(trigger=True):
    today = datetime.now(ET).date()
    rows = []
    # 09:30 open 100; prior close 100 (from get_bars 1d stub) -> gap 0, armed
    for i, (hh, mm, o, h, l, c) in enumerate([
        (9, 30, 100.0, 100.2, 99.9, 100.1),
        (9, 35, 100.1, 100.4, 100.0, 100.3),
        (10, 0, 100.3, 101.7, 100.3, 101.6 if trigger else 100.9),
        (10, 5, 101.6, 101.7, 100.8, 101.0),
        (15, 55, 101.0, 101.1, 100.9, 101.0),
        (16, 0, 101.0, 101.0, 101.0, 101.0),  # forming bar, dropped
    ]):
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 100})
    idx = [datetime(today.year, today.month, today.day, hh, mm, tzinfo=ET)
           for hh, mm, *_ in [(9, 30), (9, 35), (10, 0), (10, 5), (15, 55), (16, 0)]]
    return pd.DataFrame(rows, index=idx)


def make_runner(bars, tmp_path, dry_run=False):
    cfg = OilFadeConfig()
    broker = FakeBroker(bars)
    risk = RiskManager(starting_equity=100000.0, risk_per_trade_pct=1.0,
                       max_open_positions=1, max_position_pct=150.0, stop_loss_pct=cfg.stop_pct)
    return OilFadePaperRunner(cfg, broker, risk, tmp_path / "state.json",
                              tmp_path / "ledger.csv", dry_run=dry_run), broker


def test_trigger_entry_and_time_exit(tmp_path):
    runner, broker = make_runner(make_bars(trigger=True), tmp_path)
    assert runner.run_once() == "ok"
    sides = [o.side for o in broker.orders]
    assert sides == ["sell", "buy"], sides  # short then cover
    assert broker.orders[0].qty > 0
    # risk sizing: 1% of 100k / (0.75% * 101.6) ≈ 1312 shares
    assert 1000 < broker.orders[0].qty < 1600


def test_no_trigger_no_orders(tmp_path):
    runner, broker = make_runner(make_bars(trigger=False), tmp_path)
    assert runner.run_once() == "ok"
    assert broker.orders == []


def test_kill_switch_flattens(tmp_path):
    bars = make_bars(trigger=True).iloc[:-2]  # still short: no exit bar seen yet
    runner, broker = make_runner(bars, tmp_path)
    runner.run_once()
    assert broker.pos_qty != 0
    (tmp_path / "KILL").write_text("stop")
    runner.kill_path = tmp_path / "KILL"
    assert runner.run_once() == "killed"
    assert broker.pos_qty == 0


def test_dry_run_places_nothing(tmp_path):
    runner, broker = make_runner(make_bars(trigger=True), tmp_path, dry_run=True)
    assert runner.run_once() == "ok"
    assert broker.orders == []
    assert (tmp_path / "ledger.csv").exists()


def _one_daily_broker(bars, tmp_path, **kw):
    runner, broker = make_runner(bars, tmp_path, **kw)
    broker._one_daily_bar = True
    return runner, broker


def test_prior_close_falls_back_to_yfinance(monkeypatch, tmp_path):
    import bot.oil_fade.runner as rmod
    monkeypatch.setattr(rmod, "_yfinance_prior_close", lambda sym: 100.0)
    runner, broker = _one_daily_broker(make_bars(trigger=True), tmp_path)
    assert runner.run_once() == "ok"
    # session open 100 vs yfinance prior close 100 -> gap 0 -> armed, trades fire
    assert runner.strat.armed is True
    assert runner._gap_source == "yfinance"
    assert [o.side for o in broker.orders] == ["sell", "buy"]
    assert (tmp_path / "prior_close_cache.json").exists()


def test_yfinance_gap_up_day_stays_disarmed(monkeypatch, tmp_path):
    import bot.oil_fade.runner as rmod
    monkeypatch.setattr(rmod, "_yfinance_prior_close", lambda sym: 98.0)
    runner, broker = _one_daily_broker(make_bars(trigger=True), tmp_path)
    assert runner.run_once() == "ok"
    # gap +2.04% > 0.5% -> disarmed, no orders despite the trigger bar
    assert runner.strat.armed is False
    assert broker.orders == []


def test_yfinance_failure_disarms_fail_safe(monkeypatch, tmp_path):
    import bot.oil_fade.runner as rmod
    monkeypatch.setattr(rmod, "_yfinance_prior_close", lambda sym: None)
    runner, broker = _one_daily_broker(make_bars(trigger=True), tmp_path)
    assert runner.run_once() == "ok"
    assert runner.strat.armed is False
    assert runner.strat.gap_pct is None
    assert broker.orders == []
