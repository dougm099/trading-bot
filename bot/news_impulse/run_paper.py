"""Paper runner: poll -> detect -> verify -> impulse -> log. NEVER places orders.

Enable with NEWS_IMPULSE_ENABLED=true. Refuses to start if
LIVE_TRADING_ENABLED=true (defense in depth: this engine is pre-validation).

v1 price feed: none attached by default (NullPriceFeed). Events are detected,
verified and logged to events_log, and the impulse engine is exercised in
selftest.py. Attach a real minute-bar feed (e.g. Alpaca paper websocket) by
implementing PriceFeed and passing it to run(); only then are signals emitted,
and they are still only LOGGED, never sent to a broker.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bot.news_impulse.collectors.apewisdom import ApeWisdomCollector
from bot.news_impulse.collectors.arctic_shift import ArcticShiftCollector
from bot.news_impulse.collectors.edgar import EdgarCollector
from bot.news_impulse.collectors.fourchan import BizCollector
from bot.news_impulse.collectors.reddit import RedditCollector
from bot.news_impulse.collectors.rss import HeadlinesArchiveCollector
from bot.news_impulse.collectors.stocktwits import StocktwitsCollector
from bot.news_impulse.collectors.x_search import XCollector
from bot.news_impulse.config import NewsConfig
from bot.news_impulse.events import NewsEvent, RawPost, VelocityTracker, classify
from bot.news_impulse.impulse import ImpulseEngine
from bot.news_impulse.verify import VerificationGate


class PriceFeed:
    """Implement for live minute data. Default: no feed (detection-only mode)."""
    def minute_close(self, symbol: str) -> Optional[float]:
        return None

    def spread_bps(self, symbol: str) -> Optional[float]:
        return None


def _ensure_csv(path: str, header: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def run(cfg: NewsConfig, feed: Optional[PriceFeed] = None,
        max_iters: Optional[int] = None) -> None:
    if os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true":
        raise SystemExit("REFUSED: news-impulse will not run with LIVE_TRADING_ENABLED=true")
    if not cfg.enabled:
        raise SystemExit("Set NEWS_IMPULSE_ENABLED=true to run (paper only).")

    feed = feed or PriceFeed()
    collectors = [StocktwitsCollector(cfg.watchlist, cfg.poll_seconds),
                  EdgarCollector(cfg.ciks),
                  HeadlinesArchiveCollector(catch_up=True),
                  RedditCollector(cfg.watchlist),
                  ArcticShiftCollector(cfg.watchlist),
                  BizCollector(cfg.watchlist),
                  ApeWisdomCollector(cfg.watchlist)]
    xc = XCollector(cfg.watchlist)
    if xc.available:
        collectors.append(xc)
    else:
        print("[news-impulse] no X_BEARER_TOKEN: X collector disabled", flush=True)

    tracker = VelocityTracker(cfg.velocity_bucket_min, cfg.velocity_baseline_min, cfg.velocity_z)
    gate = VerificationGate(cfg.verify_window_min, cfg.verify_min_score)
    engine = ImpulseEngine(cfg.impulse_mult, cfg.max_wait_min, cfg.hold_min)

    open_events: Dict[str, NewsEvent] = {}  # symbol -> active unverified/verified event
    _ensure_csv(cfg.events_log, ["ts", "symbol", "category", "verified", "score", "headline", "sources"])
    _ensure_csv(cfg.signals_log, ["ts", "symbol", "action", "strength", "reason"])

    def log_event(ev: NewsEvent, verified: bool, score: float) -> None:
        with open(cfg.events_log, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), ev.symbol,
                                    ev.category, verified, round(score, 2),
                                    ev.headline, "|".join(sorted(ev.sources))])

    def log_signal(ts, symbol, action, strength, reason) -> None:
        with open(cfg.signals_log, "a", newline="") as f:
            csv.writer(f).writerow([ts.isoformat(), symbol, action, strength, reason])

    it = 0
    print(f"[news-impulse] paper mode. watchlist={cfg.watchlist}", flush=True)
    while True:
        now = datetime.now(timezone.utc)
        for coll in collectors:
            for post in coll.poll():
                sym = post.symbol
                ev = open_events.get(sym)
                cat = classify(post.text)
                spike = tracker.add(post)
                if ev is None and (cat or spike):
                    ev = NewsEvent(symbol=sym, category=cat or "velocity", first_seen=post.ts)
                    open_events[sym] = ev
                if ev is None:
                    continue
                # merge: same symbol within verify window extends the event
                if now - ev.first_seen > gate.window * 4:
                    open_events[sym] = NewsEvent(symbol=sym, category=cat or ev.category,
                                                first_seen=post.ts)
                    ev = open_events[sym]
                ev.add(post)
                score = gate.score(ev, now)
                if gate.is_verified(ev, now) and not getattr(ev, "_logged", False):
                    ev._logged = True  # type: ignore
                    log_event(ev, True, score)
                    print(f"[news-impulse] VERIFIED {sym} {ev.category} score={score:.1f} :: {ev.headline[:100]}", flush=True)
                    ref = feed.minute_close(sym)
                    if ref:
                        engine.register_event(ev, ref)
                    del open_events[sym]
        # pump impulse engine with latest minute closes
        for sym in cfg.watchlist:
            px = feed.minute_close(sym)
            if px is None:
                continue
            engine.on_minute_close(sym, px)
            for sig in engine.on_tick(sym, px, now):
                log_signal(now, sig.symbol, sig.action, sig.strength, sig.reason)
                print(f"[news-impulse] SIGNAL {sig.action} {sig.symbol} :: {sig.reason}", flush=True)

        it += 1
        if max_iters and it >= max_iters:
            break
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    run(NewsConfig())
