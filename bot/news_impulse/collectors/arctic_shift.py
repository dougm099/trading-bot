"""Arctic Shift collector: full Reddit search API (Pushshift successor).

Free, no key: https://arctic-shift.photon-reddit.com/api/posts/search
Replaces the dead Stocktwits endpoint as the primary social source: full
post bodies + selftext, sortable by new, across subs. Polite: one request
per subreddit per poll with a short pause between subs.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Set

import requests

from ..events import RawPost
from .reddit import extract_symbols

BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"
DEFAULT_SUBS = ["wallstreetbets", "stocks", "StockMarket", "investing", "options"]


def parse_post(p: dict) -> Optional[dict]:
    """Normalize one Arctic Shift post dict; None if unusable."""
    pid = p.get("id")
    title = (p.get("title") or "").strip()
    if not pid or not title:
        return None
    try:
        ts = datetime.fromtimestamp(int(p.get("created_utc", 0)), tz=timezone.utc)
    except (ValueError, TypeError):
        ts = datetime.now(timezone.utc)
    text = title + "\n" + (p.get("selftext") or "")
    return {"id": pid, "text": text, "author": p.get("author") or "unknown",
            "ts": ts, "url": "https://www.reddit.com" + (p.get("permalink") or "")}


class ArcticShiftCollector:
    def __init__(self, watchlist: List[str], subreddits: List[str] = None, limit: int = 25):
        self.watchlist = set(s.upper() for s in watchlist)
        self.subs = subreddits or DEFAULT_SUBS
        self.limit = limit
        self._seen: Set[str] = set()

    def _fetch(self, sub: str) -> list:
        try:
            r = requests.get(BASE, params={"subreddit": sub, "sort": "desc",
                                           "limit": self.limit},
                             timeout=25)
        except requests.RequestException:
            return []
        if r.status_code != 200:
            return []
        try:
            return r.json().get("data", [])
        except ValueError:
            return []

    def poll(self) -> Iterator[RawPost]:
        for sub in self.subs:
            for raw in self._fetch(sub):
                post = parse_post(raw)
                time.sleep(0.3)  # stay polite across subs
                if post is None or post["id"] in self._seen:
                    continue
                self._seen.add(post["id"])
                for sym in extract_symbols(post["text"], self.watchlist):
                    yield RawPost(symbol=sym, source="reddit",
                                  source_id=f"reddit:u/{post['author']}",
                                  text=post["text"][:500], ts=post["ts"],
                                  url=post["url"])
