"""Own-dataset builder: daily finance headlines archive.

Why: sentiment-vs-price research died for lack of a clean headline
history. RSS is free; storing one row per headline per day builds the
corpus future NLP/sentiment work needs. Cheap to keep, impossible to
backfill.

Columns: date, source, title, link. Deduped on (date, source, title).
Stdlib only (urllib + ElementTree). No feedparser dependency.
Read-only.
"""
from __future__ import annotations
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
import pandas as pd

OUT = os.path.expanduser("~/workspace/trading-bot/results/datasets/headlines.csv")
FEEDS = {
    "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
}
UA = {"User-Agent": "Mozilla/5.0 (research archive; contact: local)"}


def fetch_feed(source: str, url: str) -> list[dict]:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            xml = r.read()
        root = ET.fromstring(xml)
        items = []
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            if title:
                items.append({"date": date.today().isoformat(), "source": source,
                              "title": title[:500], "link": link[:500]})
        return items
    except Exception as e:
        print(f"  {source}: feed failed: {str(e)[:80]}")
        return []


def main():
    print(f"[{date.today()}] headlines")
    rows = []
    for source, url in FEEDS.items():
        got = fetch_feed(source, url)
        rows.extend(got)
        print(f"  {source}: {len(got)} headlines")
    if not rows:
        print("  no headlines collected")
        return
    new = pd.DataFrame(rows)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        have = set(zip(old["date"], old["source"], old["title"]))
        new = new[~new.apply(lambda r: (r["date"], r["source"], r["title"]) in have, axis=1)]
        if new.empty:
            print("  headlines already logged")
            return
        old = pd.concat([old, new], ignore_index=True)
        old.to_csv(OUT, index=False)
    else:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        new.to_csv(OUT, index=False)
    print(f"  +{len(new)} rows -> {OUT}")


if __name__ == "__main__":
    main()
