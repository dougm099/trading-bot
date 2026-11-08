"""Multi-source verification gate.

A single anonymous tweet never triggers a trade (cf. the 2022 fake Eli Lilly
tweet that moved the stock 4%+). An event is verified when its weighted
source score clears the threshold inside the verification window.

Weights: wire/outlet sources (rss:cnbc, rss:reuters, rss:yahoo, rss:marketwatch,
edgar) count 2.0. They have editorial desks. Social sources (stocktwits, x)
count 1.0 each, and multiple posts from the SAME handle/domain count once.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

from .events import NewsEvent

WIRE_SOURCES = {"rss:cnbc", "rss:reuters", "rss:yahoo", "rss:marketwatch", "edgar"}


def source_weight(source: str) -> float:
    return 2.0 if source in WIRE_SOURCES else 1.0


class VerificationGate:
    def __init__(self, window_min: int = 15, min_score: float = 2.0):
        self.window = timedelta(minutes=window_min)
        self.min_score = min_score

    def score(self, event: NewsEvent, now: datetime) -> float:
        total = 0.0
        seen = set()
        for p in event.posts:
            if now - p.ts > self.window:
                continue
            key = (p.source, p.source_id)
            if key in seen:
                continue
            seen.add(key)
            total += source_weight(p.source)
        return total

    def is_verified(self, event: NewsEvent, now: datetime) -> bool:
        return self.score(event, now) >= self.min_score
