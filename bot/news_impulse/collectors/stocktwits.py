"""Stocktwits streams collector (public endpoint, no key).

STATUS 2026-09-20: the unauthenticated v2 streams endpoint now returns 404
("Page Not Found"). Stocktwits appears to have killed public access. The
collector is kept for interface stability and fails gracefully (empty polls);
the engine runs on the remaining sources. Re-enable if authenticated access
or a working public endpoint is found.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

import requests

from ..events import RawPost

BASE = "https://stocktwits.com/api/2/streams/symbol/{sym}.json"
HEADERS = {"User-Agent": "trading-bot-research/1.0 (paper-only)"}


class StocktwitsCollector:
    def __init__(self, symbols: List[str], poll_seconds: int = 45):
        self.symbols = [s.upper() for s in symbols]
        self.poll_seconds = poll_seconds
        self._last_id: Dict[str, int] = {}
        self._backoff_until: float = 0.0

    def _fetch(self, symbol: str) -> list:
        if time.time() < self._backoff_until:
            return []
        try:
            r = requests.get(BASE.format(sym=symbol), headers=HEADERS, timeout=15)
        except requests.RequestException:
            return []
        if r.status_code == 429:
            self._backoff_until = time.time() + 300  # 5 min cool-down
            return []
        if r.status_code != 200:
            return []
        try:
            return r.json().get("messages", [])
        except ValueError:
            return []

    def poll(self) -> Iterator[RawPost]:
        for sym in self.symbols:
            for m in self._fetch(sym):
                mid = int(m.get("id", 0))
                if mid <= self._last_id.get(sym, 0):
                    continue
                self._last_id[sym] = max(self._last_id.get(sym, 0), mid)
                created = m.get("created_at", "")
                try:
                    ts = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)
                user = (m.get("user") or {}).get("username", "unknown")
                yield RawPost(symbol=sym, source="stocktwits", source_id=f"st:{user}",
                              text=m.get("body", ""), ts=ts,
                              url=f"https://stocktwits.com/{user}/post/{mid}")
