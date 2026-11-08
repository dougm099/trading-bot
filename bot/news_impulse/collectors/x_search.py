"""X (Twitter) v2 recent-search collector.

Needs X_BEARER_TOKEN in the environment. Free-tier rate limits are severe
(~1 search/min on small result sets), so this collector:
- queries one cashtag per poll tick, rotating through the watchlist,
- requests minimal fields, and
- silently skips when no token is configured (engine runs on the other
  sources instead).

X is deliberately SECONDARY in this design until a forward paper record
justifies paying for a higher tier.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List

import requests

from ..events import RawPost

SEARCH_URL = "https://api.x.com/2/tweets/search/recent"


class XCollector:
    def __init__(self, symbols: List[str]):
        self.symbols = [s.upper() for s in symbols]
        self.token = os.environ.get("X_BEARER_TOKEN", "")
        self._idx = 0
        self._since: dict = {}
        self._backoff_until: float = 0.0

    @property
    def available(self) -> bool:
        return bool(self.token)

    def poll(self) -> Iterator[RawPost]:
        if not self.available or time.time() < self._backoff_until:
            return
        sym = self.symbols[self._idx % len(self.symbols)]
        self._idx += 1
        params = {
            "query": f"${sym} -is:retweet lang:en",
            "max_results": 20,
            "tweet.fields": "created_at,author_id",
        }
        if sym in self._since:
            params["since_id"] = self._since[sym]
        try:
            r = requests.get(SEARCH_URL, params=params,
                             headers={"Authorization": f"Bearer {self.token}"},
                             timeout=15)
        except requests.RequestException:
            return
        if r.status_code == 429:
            self._backoff_until = time.time() + 900  # 15 min
            return
        if r.status_code != 200:
            return
        try:
            data = r.json().get("data", [])
        except ValueError:
            return
        for t in data:
            tid = t.get("id", "")
            self._since[sym] = tid  # ids are time-ordered; last wins
            try:
                ts = datetime.strptime(t["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                ts = datetime.now(timezone.utc)
            yield RawPost(symbol=sym, source="x",
                          source_id=f"x:{t.get('author_id', '?')}",
                          text=t.get("text", ""), ts=ts,
                          url=f"https://x.com/i/status/{tid}")
