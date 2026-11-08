"""4chan /biz/ collector: board catalog JSON (free, no auth).

https://a.4cdn.org/biz/catalog.json: all live threads' OPs. /biz/ is raw,
unfiltered meme-stock/crypto sentiment: high noise, but squeezes and
narrative flips often surface here before sanitized platforms. Catalog-only
(OPs, not replies) keeps it to one polite request per poll; 4chan asks
<=1 req/sec and we poll far slower than that.

Posts are anonymous: source_id is the thread number (one voice per thread).
HTML in post bodies is stripped. Timestamps: catalog OPs carry no unix time,
so ts = fetch time (documented limitation; fine for velocity bucketing).
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Set

import requests

from ..events import RawPost
from .reddit import extract_symbols

CATALOG = "https://a.4cdn.org/biz/catalog.json"
TAG_RE = re.compile(r"<[^>]+>")


def clean_post(sub: str, com: str) -> str:
    text = (sub + "\n" + com) if sub else com
    text = TAG_RE.sub(" ", text)
    return html.unescape(text).strip()


class BizCollector:
    def __init__(self, watchlist: List[str]):
        self.watchlist = set(s.upper() for s in watchlist)
        self._seen: Set[int] = set()

    def _fetch(self) -> list:
        try:
            r = requests.get(CATALOG, headers={"User-Agent": "research/1.0"},
                             timeout=25)
        except requests.RequestException:
            return []
        if r.status_code != 200:
            return []
        try:
            return r.json()
        except ValueError:
            return []

    def poll(self) -> Iterator[RawPost]:
        now = datetime.now(timezone.utc)
        for page in self._fetch():
            for t in page.get("threads", []):
                no = t.get("no")
                if not no or no in self._seen:
                    continue
                self._seen.add(no)
                text = clean_post(t.get("sub") or "", t.get("com") or "")
                if not text:
                    continue
                for sym in extract_symbols(text, self.watchlist):
                    yield RawPost(symbol=sym, source="4chan",
                                  source_id=f"4chan:{no}", text=text[:500],
                                  ts=now,
                                  url=f"https://boards.4chan.org/biz/thread/{no}")
