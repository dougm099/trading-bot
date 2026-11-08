"""Market-hawk intraday collector: snapshots + wide news, market-hours gated.

Runs hourly (via cron). No-ops outside US equity market hours
(Mon-Fri 06:25-13:10 PT) to avoid burning API calls on dead air.

Writes:
  results/datasets/hawk_snapshots.csv  ts,symbol,price,prev_close,pct_change
  results/datasets/news_wide.csv        ts,date,source,category,title,link

Watchlist spans equities, sectors, vol, rates, credit, dollar, gold,
oil, foreign markets, and crypto -- the cross-asset board the briefing
reads for correlation shifts. Read-only.
"""
from __future__ import annotations
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PT = ZoneInfo("America/Los_Angeles")
DATA = os.path.expanduser("~/workspace/trading-bot/results/datasets")
SNAP = os.path.join(DATA, "hawk_snapshots.csv")
NEWS = os.path.join(DATA, "news_wide.csv")

WATCHLIST = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY",
    "^VIX", "^TNX", "^FVX",
    "GLD", "USO", "UUP", "TLT", "HYG",
    "EFA", "EEM",
    "BTC-USD", "ETH-USD",
    # user's own stocks (added 2026-09-21)
    "AAPL", "PLTR", "HIMS", "RCAT",
]

FEEDS = {
    # source: (url, category)
    "bbc_world": ("https://feeds.bbci.co.uk/news/world/rss.xml", "world"),
    "npr": ("https://feeds.npr.org/1001/rss.xml", "us"),
    "aljazeera": ("https://www.aljazeera.com/xml/rss/all.xml", "world"),
    "guardian_world": ("https://www.theguardian.com/world/rss", "world"),
    "ars_technica": ("http://feeds.arstechnica.com/arstechnica/index", "tech"),
    "coindesk": ("https://www.coindesk.com/arc/outboundfeeds/rss", "crypto"),
    "cnbc": ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "finance"),
    "marketwatch": ("https://feeds.marketwatch.com/marketwatch/topstories/", "finance"),
}
UA = {"User-Agent": "Mozilla/5.0 (research archive; contact: local)"}


def market_open_now() -> bool:
    now = datetime.now(PT)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 6 * 60 + 25 <= mins <= 13 * 60 + 10


def collect_snapshots() -> None:
    now = datetime.now(PT)
    ts = now.strftime("%Y-%m-%dT%H:%M")
    print(f"[{ts} PT] hawk snapshots")
    data = yf.download(WATCHLIST, period="5d", interval="1d",
                       auto_adjust=True, progress=False)["Close"]
    rows = []
    for sym in WATCHLIST:
        try:
            s = data[sym].dropna()
            if len(s) < 2:
                continue
            price, prev = float(s.iloc[-1]), float(s.iloc[-2])
            rows.append({"ts": ts, "symbol": sym, "price": round(price, 4),
                         "prev_close": round(prev, 4),
                         "pct_change": round((price / prev - 1) * 100, 3)})
        except Exception as e:
            print(f"  {sym}: {str(e)[:60]}")
    if not rows:
        print("  no snapshot rows")
        return
    new = pd.DataFrame(rows)
    if os.path.exists(SNAP):
        old = pd.read_csv(SNAP)
        have = set(zip(old["ts"], old["symbol"]))
        new = new[~new.apply(lambda r: (r["ts"], r["symbol"]) in have, axis=1)]
    if len(new):
        new.to_csv(SNAP, mode="a", header=not os.path.exists(SNAP), index=False)
    print(f"  wrote {len(new)} snapshot rows")


def collect_news() -> None:
    now = datetime.now(PT)
    ts = now.strftime("%Y-%m-%dT%H:%M")
    today = now.strftime("%Y-%m-%d")
    print(f"[{ts} PT] wide news")
    rows = []
    for source, (url, cat) in FEEDS.items():
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                root = ET.fromstring(r.read())
            n = 0
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                if title:
                    rows.append({"ts": ts, "date": today, "source": source,
                                 "category": cat, "title": title[:500],
                                 "link": link[:500]})
                    n += 1
            print(f"  {source} ({cat}): {n}")
        except Exception as e:
            print(f"  {source}: failed: {str(e)[:70]}")
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(NEWS):
        old = pd.read_csv(NEWS)
        have = set(zip(old["source"], old["title"]))
        new = new[~new.apply(lambda r: (r["source"], r["title"]) in have, axis=1)]
    if len(new):
        new.to_csv(NEWS, mode="a", header=not os.path.exists(NEWS), index=False)
    print(f"  wrote {len(new)} news rows")


def main() -> None:
    if not market_open_now():
        print("market closed -- hawk resting")
        return
    collect_snapshots()
    collect_news()
    print("hawk snapshot done")


if __name__ == "__main__":
    main()
