"""ApeWisdom collector: WSB ticker attention rankings (free, no key).

https://apewisdom.io/api/v1.0/filter/all-stocks/ ranks tickers by 24h
wallstreetbets mentions/upvotes. This collector emits one synthetic post per
watchlist ticker whose rank JUMPED into the top N vs 24h ago. An
attention spike, which is the meme-stock-relevant signal (absolute rank is
just background). Deduped per ticker per day; the feed updates slowly so
repeat polls are cheap no-ops.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterator, List, Set

import requests

from ..events import RawPost

URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/"


def rank_spikes(results: List[dict], watchlist: Set[str], top_n: int = 50,
                min_jump: int = 5) -> List[dict]:
    """Return watchlist tickers that jumped >= min_jump spots into top_n."""
    out = []
    for r in results:
        t = (r.get("ticker") or "").upper()
        if t not in watchlist:
            continue
        rank, prev = r.get("rank"), r.get("rank_24h_ago")
        if not rank or not prev or rank > top_n:
            continue
        if prev - rank >= min_jump:
            out.append(r)
    return out


class ApeWisdomCollector:
    def __init__(self, watchlist: List[str], top_n: int = 50, min_jump: int = 5):
        self.watchlist = set(s.upper() for s in watchlist)
        self.top_n = top_n
        self.min_jump = min_jump
        self._seen: Set[str] = set()  # ticker+date

    def _fetch(self) -> list:
        try:
            r = requests.get(URL, timeout=25)
        except requests.RequestException:
            return []
        if r.status_code != 200:
            return []
        try:
            return r.json().get("results", [])
        except ValueError:
            return []

    def poll(self) -> Iterator[RawPost]:
        now = datetime.now(timezone.utc)
        day = now.date().isoformat()
        for r in rank_spikes(self._fetch(), self.watchlist, self.top_n, self.min_jump):
            t = r["ticker"].upper()
            key = f"{t}:{day}"
            if key in self._seen:
                continue
            self._seen.add(key)
            yield RawPost(
                symbol=t, source="apewisdom", source_id="apewisdom:rankings",
                text=(f"WSB attention spike: ${t} rank #{r['rank']} (was #{r['rank_24h_ago']} 24h ago), "
                      f"{r.get('mentions', '?')} mentions, {r.get('upvotes', '?')} upvotes"),
                ts=now, url="https://apewisdom.io/")
