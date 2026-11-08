"""Backfill: replay the headlines archive through event detection.

headlines.csv has date (not timestamp) granularity, so velocity spikes are
out of scope here. This replays the taxonomy + verification path only:
attribute symbols -> classify -> group by (symbol, date) -> score with a
24h-window VerificationGate -> log verified events.

Output: results/news_impulse_events_backfill.csv
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.news_impulse.collectors.rss import HeadlinesArchiveCollector, ARCHIVE
from bot.news_impulse.events import NewsEvent, classify
from bot.news_impulse.verify import VerificationGate

OUT = os.path.expanduser("~/workspace/trading-bot/results/news_impulse_events_backfill.csv")


def main() -> None:
    coll = HeadlinesArchiveCollector(path=ARCHIVE, catch_up=False)
    groups = defaultdict(list)  # (symbol, date) -> [RawPost]
    n_posts = 0
    for p in coll.poll():
        n_posts += 1
        groups[(p.symbol, p.ts.date())].append(p)

    gate = VerificationGate(window_min=24 * 60, min_score=2.0)
    now = datetime.now(timezone.utc)
    verified = []
    for (sym, day), posts in sorted(groups.items()):
        cat = next((c for p in posts if (c := classify(p.text))), "general")
        ev = NewsEvent(symbol=sym, category=cat, first_seen=posts[0].ts)
        for p in posts:
            ev.add(p)
        score = gate.score(ev, now)
        if gate.is_verified(ev, now):
            verified.append((day.isoformat(), sym, cat, round(score, 2),
                             "|".join(sorted(ev.sources)), ev.headline))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "symbol", "category", "score", "sources", "headline"])
        w.writerows(verified)

    by_cat = defaultdict(int)
    for v in verified:
        by_cat[v[2]] += 1
    print(f"posts attributed: {n_posts} | symbol-days: {len(groups)} | verified events: {len(verified)}")
    print("by category:", dict(by_cat))
    for v in verified[:15]:
        print(f"  {v[0]} {v[1]:5s} {v[2]:9s} score={v[3]} :: {v[5][:90]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
