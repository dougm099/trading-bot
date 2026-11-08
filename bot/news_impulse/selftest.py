"""Deterministic self-test for the news-impulse engine (no network).

Covers: velocity spike detection, taxonomy classification, verification
gate weights, impulse entry/exit logic, and the event risk halt.
Run: python -m bot.news_impulse.selftest
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bot.news_impulse.events import NewsEvent, RawPost, VelocityTracker, classify
from bot.news_impulse.impulse import ImpulseEngine
from bot.news_impulse.risk import EventRiskManager
from bot.news_impulse.verify import VerificationGate

NOW = datetime.now(timezone.utc)


def post(sym, src, sid, text, age_min=0):
    return RawPost(symbol=sym, source=src, source_id=sid, text=text,
                   ts=NOW - timedelta(minutes=age_min))


def test_taxonomy():
    assert classify("FDA approves new Alzheimer's drug") == "FDA"
    assert classify("Company to acquire rival for $2B in merger") == "MERGER"
    assert classify("CEO resigns effective immediately") == "CEO"
    assert classify("just bought more calls, going to the moon") is None
    print("taxonomy ok")


def test_velocity():
    vt = VelocityTracker(bucket_min=5, baseline_min=120, z=3.0)
    # quiet baseline: 1 post per bucket over 24 buckets
    for i in range(24):
        p = post("SPY", "stocktwits", f"st:u{i}", "boring", age_min=120 - i * 5)
        assert vt.add(p) is False
    # spike: 12 posts in the current bucket
    spiked = False
    for i in range(12):
        spiked = vt.add(post("SPY", "stocktwits", f"st:s{i}", "BREAKING news!!", age_min=1))
    assert spiked, "velocity spike should fire"
    print("velocity ok")


def test_verify():
    gate = VerificationGate(window_min=15, min_score=2.0)
    ev = NewsEvent(symbol="AAPL", category="MERGER", first_seen=NOW)
    ev.add(post("AAPL", "stocktwits", "st:joe", "acquiring XYZ", age_min=2))
    assert not gate.is_verified(ev, NOW), "single social post must not verify"
    ev.add(post("AAPL", "x", "x:jane", "acquiring XYZ confirmed?", age_min=1))
    assert gate.is_verified(ev, NOW), "two independent social sources verify"
    ev2 = NewsEvent(symbol="AAPL", category="MERGER", first_seen=NOW)
    ev2.add(post("AAPL", "rss:cnbc", "cnbc.com", "Apple acquiring XYZ", age_min=2))
    assert gate.is_verified(ev2, NOW), "single wire source verifies (weight 2)"
    print("verify ok")


def test_impulse():
    eng = ImpulseEngine(impulse_mult=2.0, max_wait_min=10, hold_min=10)
    # build minute history: tiny noise around 100 so baseline vol > 0
    px = 100.0
    for i in range(12):
        px += 0.05 if i % 2 == 0 else -0.05
        eng.on_minute_close("NVDA", px)
    vol = eng.realized_vol_per_min("NVDA")
    ev = NewsEvent(symbol="NVDA", category="FDA", first_seen=NOW)
    eng.register_event(ev, 100.0)
    # small wiggle: no entry
    assert eng.on_tick("NVDA", 100.05, NOW + timedelta(minutes=1)) == []
    # big impulse up: 2%+ move vs ~0 vol -> entry
    sigs = eng.on_tick("NVDA", 102.5, NOW + timedelta(minutes=2))
    assert len(sigs) == 1 and sigs[0].action == "buy", sigs
    # time stop after 10 min
    sigs = eng.on_tick("NVDA", 103.0, NOW + timedelta(minutes=12))
    assert len(sigs) == 1 and sigs[0].action == "flat" and "time-stop" in sigs[0].reason
    print("impulse ok")


def test_impulse_stop():
    eng = ImpulseEngine(impulse_mult=2.0, max_wait_min=10, hold_min=60)
    px = 100.0
    for i in range(12):
        px += 0.05 if i % 2 == 0 else -0.05
        eng.on_minute_close("AMD", px)
    ev = NewsEvent(symbol="AMD", category="LAWSUIT", first_seen=NOW)
    eng.register_event(ev, 100.0)
    sigs = eng.on_tick("AMD", 98.0, NOW + timedelta(minutes=1))  # impulse down 2.0
    assert sigs and sigs[0].action == "sell"
    # hard stop at 0.5x impulse against = 99.0
    sigs = eng.on_tick("AMD", 99.5, NOW + timedelta(minutes=2))
    assert sigs and sigs[0].action == "flat" and "stop" in sigs[0].reason, sigs
    print("impulse stop ok")


def test_risk_halt():
    from datetime import date
    rm = EventRiskManager(equity=10_000, risk_per_event_pct=0.5,
                          max_concurrent=1, max_daily_loss_pct=2.0)
    today = date.today()
    assert rm.can_open("SPY", today, 10_000)
    rm.register_open("SPY")
    assert not rm.can_open("QQQ", today, 10_000), "concurrency cap"
    rm.register_close("SPY", -250.0, today, 9_750)  # -2.5% day
    assert rm.halted and not rm.can_open("QQQ", today, 9_750), "daily halt"
    assert rm.size_qty(100.0, 2.0) == 25.0  # $50 risk / $2 stop
    print("risk ok")


def test_attribution():
    from bot.news_impulse.collectors.rss import attribute_symbols
    assert attribute_symbols("Apple unveils new iPhone") == ["AAPL"]
    assert attribute_symbols("Nvidia CEO Jensen Huang speaks") == ["NVDA"]
    assert attribute_symbols("S&P 500 rallies as Wall Street cheers") == ["SPY"]
    assert attribute_symbols("Bessent meets China Vice Premier") == []
    print("attribution ok")


def test_reddit_symbols():
    from bot.news_impulse.collectors.reddit import extract_symbols
    wl = {"NVDA", "AAPL", "SPY", "TSLA"}
    assert extract_symbols("$NVDA to the moon", wl) == ["NVDA"]
    assert extract_symbols("$nvda calls printing", wl) == ["NVDA"]  # cashtags case-insensitive
    assert extract_symbols("Apple earnings beat", wl) == ["AAPL"]
    assert extract_symbols("$GME squeeze", wl) == []  # not on watchlist
    assert extract_symbols("market rallies broadly", wl) == []
    print("reddit symbols ok")


def test_meme_collectors():
    from bot.news_impulse.collectors.arctic_shift import parse_post
    from bot.news_impulse.collectors.fourchan import clean_post
    from bot.news_impulse.collectors.apewisdom import rank_spikes
    p = parse_post({"id": "abc123", "title": "$NVDA earnings tomorrow",
                    "selftext": "loading calls", "author": "ape1",
                    "created_utc": 1789946223, "permalink": "/r/wsb/x"})
    assert p and p["author"] == "ape1" and p["url"].endswith("/r/wsb/x")
    assert parse_post({"id": "", "title": "x"}) is None
    assert clean_post("ticker thread", "&gt;$GME<br>to the moon") == "ticker thread\n>$GME to the moon"
    wl = {"NVDA", "GME"}
    res = [{"ticker": "NVDA", "rank": 8, "rank_24h_ago": 30, "mentions": 40, "upvotes": 100},
           {"ticker": "GME", "rank": 3, "rank_24h_ago": 4, "mentions": 90, "upvotes": 500},
           {"ticker": "AAPL", "rank": 1, "rank_24h_ago": 50, "mentions": 5, "upvotes": 5}]
    spikes = rank_spikes(res, wl)
    assert [r["ticker"] for r in spikes] == ["NVDA"], spikes  # GME jump too small, AAPL off-watchlist
    print("meme collectors ok")


if __name__ == "__main__":
    test_meme_collectors()
    test_reddit_symbols()
    test_attribution()
    test_taxonomy()
    test_velocity()
    test_verify()
    test_impulse()
    test_impulse_stop()
    test_risk_halt()
    print("ALL SELFTESTS PASSED")
