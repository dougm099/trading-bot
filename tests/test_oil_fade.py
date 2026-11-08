"""Unit tests for the oil-fade strategy core (synthetic bars, no network)."""
from datetime import datetime

import pytest

from bot.oil_fade.strategy import Bar, OilFadeConfig, OilFadeStrategy


def bar(ts, o, h, l, c):
    return Bar(datetime(2026, 9, 28, *ts), o, h, l, c)


def test_gap_up_day_not_armed():
    s = OilFadeStrategy(OilFadeConfig())
    s.new_day(session_open=100.0, prior_close=98.0)   # +2.04% gap
    assert s.armed is False
    sig = s.on_bar(bar((10, 0), 100, 102, 100, 101.8))  # +1.8% close, would trigger
    assert sig.action == "none"


def test_gap_down_day_triggers():
    s = OilFadeStrategy(OilFadeConfig())
    s.new_day(session_open=100.0, prior_close=102.0)  # -1.96% gap
    assert s.armed is True
    sig = s.on_bar(bar((10, 0), 100, 101.6, 100, 101.6))
    assert sig.action == "short" and sig.reason == "trigger"
    assert s.entry == 101.6 and abs(s.stop - 101.6 * 1.0075) < 1e-9


def test_stop_fires_on_touch():
    s = OilFadeStrategy(OilFadeConfig())
    s.new_day(session_open=100.0, prior_close=100.0)
    s.on_bar(bar((10, 0), 100, 101.6, 100, 101.6))
    sig = s.on_bar(bar((10, 5), 101.6, 102.5, 101.5, 101.7))  # high 102.5 > stop 102.362
    assert sig.action == "cover" and sig.reason == "stop"
    assert s.is_flat


def test_time_exit_covers():
    s = OilFadeStrategy(OilFadeConfig())
    s.new_day(session_open=100.0, prior_close=100.0)
    s.on_bar(bar((10, 0), 100, 101.6, 100, 101.6))
    sig = s.on_bar(bar((15, 55), 100.5, 100.6, 100.4, 100.5))
    assert sig.action == "cover" and sig.reason == "time_exit"


def test_no_trigger_stays_flat():
    s = OilFadeStrategy(OilFadeConfig())
    s.new_day(session_open=100.0, prior_close=100.0)
    for hh, mm in [(9, 35), (10, 0), (15, 55)]:
        sig = s.on_bar(bar((hh, mm), 100, 100.5, 99.8, 100.2))
        assert sig.action == "none"
    assert s.is_flat


def test_one_trade_per_day():
    s = OilFadeStrategy(OilFadeConfig(exit_time_et="10:30"))
    s.new_day(session_open=100.0, prior_close=100.0)
    s.on_bar(bar((10, 0), 100, 101.6, 100, 101.6))     # short
    s.on_bar(bar((10, 30), 100.5, 100.6, 100.4, 100.5))  # covered by time exit
    sig = s.on_bar(bar((11, 0), 100, 102.0, 100, 101.9))  # +1.9% again
    assert sig.action == "none", "must not re-enter after exit"


def test_no_gap_filter_model_d():
    s = OilFadeStrategy(OilFadeConfig(gap_max_pct=None))
    s.new_day(session_open=100.0, prior_close=90.0)  # huge gap up
    assert s.armed is True
    sig = s.on_bar(bar((10, 0), 100, 101.6, 100, 101.6))
    assert sig.action == "short"


def test_unknown_prior_close_disarms_d2():
    # Fail safe: no prior close -> gap unmeasurable -> stay disarmed so we
    # never trade blind outside the frozen D2 prereg.
    s = OilFadeStrategy(OilFadeConfig())  # D2, gap_max_pct=0.5
    s.new_day(session_open=100.0, prior_close=None)
    assert s.gap_pct is None
    assert s.armed is False
    sig = s.on_bar(bar((10, 0), 100, 101.6, 100, 101.6))  # +1.6% trigger bar
    assert sig.action == "none", "must not trigger while disarmed"


def test_unknown_prior_close_still_arms_no_filter_model_d():
    # Plain Model D (no gap filter) does not depend on prior close.
    s = OilFadeStrategy(OilFadeConfig(gap_max_pct=None))
    s.new_day(session_open=100.0, prior_close=None)
    assert s.armed is True
