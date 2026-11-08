"""Reddit collector: per-subreddit /new/.rss Atom feeds (no key, no OAuth).

Why RSS: reddit's .json endpoints block scripted access, pullpush.io blocks
agents, and the OAuth API needs user-created app credentials. The .rss feeds
still serve ~25 newest posts per subreddit anonymously: enough for
event-velocity detection on meme-stock subs.

Polls r/wallstreetbets, r/stocks, r/StockMarket, r/investing, r/options.
Symbols come from cashtags ($NVDA) first, falling back to company-name
matching (bot.news_impulse.collectors.rss). Only watchlist symbols are
yielded. source weight is social (1.0). Reddit never verifies alone.
"""
from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Set

from ..events import RawPost
from .rss import attribute_symbols

NS = {"a": "http://www.w3.org/2005/Atom"}
UA = {"User-Agent": "Mozilla/5.0 (research archive; contact: local)"}

DEFAULT_SUBS = ["wallstreetbets", "stocks", "StockMarket", "investing", "options"]
CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")


def extract_symbols(title: str, watchlist: Set[str]) -> List[str]:
    """Cashtags first ($NVDA, case-insensitive), then company-name attribution.
    Watchlist-only."""
    tags = [t.upper() for t in CASHTAG.findall(title) if t.upper() in watchlist]
    if tags:
        # dedup, preserve order
        return list(dict.fromkeys(tags))
    return [s for s in attribute_symbols(title) if s in watchlist]


class RedditCollector:
    def __init__(self, watchlist: List[str], subreddits: List[str] = None):
        self.watchlist = set(s.upper() for s in watchlist)
        self.subs = subreddits or DEFAULT_SUBS
        self._seen: Set[str] = set()

    def _fetch(self, sub: str) -> list:
        # NOTE: python-requests gets its connection RST mid-read by reddit's
        # CDN (TLS fingerprinting); curl is reliable, so we shell out to it.
        url = f"https://www.reddit.com/r/{sub}/new/.rss"
        for attempt in range(2):
            try:
                proc = subprocess.run(
                    ["curl", "-s", "-m", "25", "-A", UA["User-Agent"], url],
                    capture_output=True, timeout=30)
            except (subprocess.SubprocessError, OSError):
                continue
            if proc.returncode != 0 or not proc.stdout:
                continue
            try:
                root = ET.fromstring(proc.stdout)
            except ET.ParseError:
                return []
            return root.findall("a:entry", NS)
        return []

    def poll(self) -> Iterator[RawPost]:
        for sub in self.subs:
            for entry in self._fetch(sub):
                link = entry.find("a:link", NS)
                href = link.get("href", "") if link is not None else ""
                if not href or href in self._seen:
                    continue
                self._seen.add(href)
                title = (entry.findtext("a:title", default="", namespaces=NS) or "").strip()
                author_el = entry.find("a:author/a:name", NS)
                author = (author_el.text if author_el is not None else "unknown").strip("/u ")
                updated = entry.findtext("a:updated", default="", namespaces=NS) or ""
                try:
                    ts = datetime.strptime(updated[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    ts = datetime.now(timezone.utc)
                syms = extract_symbols(title, self.watchlist)
                for sym in syms:
                    yield RawPost(symbol=sym, source="reddit",
                                  source_id=f"reddit:u/{author}", text=title,
                                  ts=ts, url=href)
