"""EDGAR 8-K watcher: per-company Atom feeds for the watchlist.

8-K filings (mergers, CEO changes, bankruptcies, material agreements) are
among the few news events with documented short-horizon tradability. Free,
no key; SEC requires a descriptive User-Agent. Set SEC_USER_AGENT
(e.g. "YourName you@example.com") or the default research UA is used.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, Iterator, List

import requests

from ..events import RawPost

ATOM = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=10&output=atom"
NS = {"a": "http://www.w3.org/2005/Atom"}


class EdgarCollector:
    def __init__(self, ciks: Dict[str, str]):
        self.ciks = {s.upper(): c for s, c in ciks.items()}
        self._seen: Dict[str, set] = {s: set() for s in self.ciks}
        ua = os.environ.get("SEC_USER_AGENT", "trading-bot-research/1.0 (paper-only; contact: research@localhost)")
        self.headers = {"User-Agent": ua, "Accept-Encoding": "gzip"}

    def poll(self) -> Iterator[RawPost]:
        for sym, cik in self.ciks.items():
            try:
                r = requests.get(ATOM.format(cik=cik), headers=self.headers, timeout=20)
            except requests.RequestException:
                continue
            if r.status_code != 200:
                continue
            try:
                root = ET.fromstring(r.content)
            except ET.ParseError:
                continue
            for entry in root.findall("a:entry", NS):
                link = entry.find("a:link", NS)
                href = link.get("href", "") if link is not None else ""
                if not href or href in self._seen[sym]:
                    continue
                self._seen[sym].add(href)
                title = (entry.findtext("a:title", default="", namespaces=NS) or "").strip()
                updated = entry.findtext("a:updated", default="", namespaces=NS) or ""
                try:
                    ts = datetime.strptime(updated[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    ts = datetime.now(timezone.utc)
                yield RawPost(symbol=sym, source="edgar", source_id="edgar:8-K",
                              text=f"8-K filed: {title}", ts=ts, url=href)
