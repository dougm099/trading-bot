"""RSS headlines collector: tails results/datasets/headlines.csv.

The daily cron (scripts/collect_headlines.py) appends wire headlines; this
collector turns new rows into RawPost objects with source weight 2.0 (wire).
Headlines carry no cashtags, so symbols are attributed by company/index name
matching: specific companies first, broad-market terms fall through to SPY.
A headline can attribute to multiple symbols; each symbol's event is
evaluated independently downstream.
"""
from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

from ..events import RawPost

ARCHIVE = os.path.expanduser("~/workspace/trading-bot/results/datasets/headlines.csv")

# symbol -> regexes, checked in dict order (specific before broad)
ATTRIBUTION: Dict[str, List[str]] = {
    "AAPL": [r"\bapple\b"],
    "MSFT": [r"\bmicrosoft\b"],
    "NVDA": [r"\bnvidia\b"],
    "TSLA": [r"\btesla\b", r"\bmusk\b"],
    "META": [r"\bmeta\b", r"\bfacebook\b"],
    "AMZN": [r"\bamazon\b"],
    "GOOGL": [r"\bgoogle\b", r"\balphabet\b"],
    "AMD": [r"\bamd\b"],
    "JPM": [r"\bjpmorgan\b", r"\bj\.?p\.? morgan\b"],
    "NFLX": [r"\bnetflix\b"],
    "QQQ": [r"\bnasdaq\b"],
    "IWM": [r"\brussell 2000\b", r"\bsmall-?cap(s)?\b"],
    "SPY": [r"s&p ?500", r"\bs&p\b", r"wall street", r"\bu\.?s\.? stocks\b",
            r"\bstock market\b", r"\bstocks\b.*\b(rally|fall|drop|rise|slide|jump|tumble)\b"],
}
_COMPILED = {s: [re.compile(p, re.IGNORECASE) for p in pats] for s, pats in ATTRIBUTION.items()}


def attribute_symbols(title: str) -> List[str]:
    hits = [s for s, pats in _COMPILED.items() if any(p.search(title) for p in pats)]
    # broad-market SPY tag only if nothing more specific matched
    if hits and hits != ["SPY"]:
        hits = [h for h in hits if h != "SPY"] or hits
    return hits


class HeadlinesArchiveCollector:
    """Yields RawPost for archive rows. catch_up=True (default) marks existing
    rows seen on first poll so restarts don't re-emit history."""

    def __init__(self, path: str = ARCHIVE, catch_up: bool = True):
        self.path = path
        self._seen_links = set()
        if catch_up:
            self._scan(mark_only=True)

    def _scan(self, mark_only: bool = False) -> Iterator[RawPost]:
        if not os.path.exists(self.path):
            return
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                link = row.get("link", "")
                if not link or link in self._seen_links:
                    continue
                self._seen_links.add(link)
                if mark_only:
                    continue
                title = row.get("title", "")
                source = row.get("source", "rss")
                try:
                    ts = datetime.strptime(row.get("date", ""), "%Y-%m-%d").replace(
                        tzinfo=timezone.utc)
                except ValueError:
                    ts = datetime.now(timezone.utc)
                for sym in attribute_symbols(title):
                    yield RawPost(symbol=sym, source=f"rss:{source}",
                                  source_id=f"rss:{source}", text=title,
                                  ts=ts, url=link)

    def poll(self) -> Iterator[RawPost]:
        yield from self._scan()
