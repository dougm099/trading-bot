"""Real-time news wire for tradeable headlines.

Polls major RSS feeds and fires an ALERT for fresh, high-impact headlines:
  TIER 1 GEOPOLITICAL -- escalation events (strikes, Hormuz, new sanctions...)
  TIER 2 POLICY SHOCK -- Trump/Fed/tariff actions that reprice markets
  TIER 3 OFF-RAMP     -- de-escalation (ceasefire, roadmap to end it, Hormuz
                         reopening, sanctions relief...). This is the
                         fade-the-risk-premium moment: as soon as the other
                         side signals openness to ending it, the alert fires
                         and the clock-in runs the news+volume analysis.

Designed to run every 5 minutes via cron. Only headlines published within the last
FRESH_MIN minutes are eligible, and each headline alerts at most once
(deduped in news_wire_seen.json).

Every alert is also appended to news_wire_alerts.csv with a timestamp --
that log is the raw material for a prospective study of whether SPY actually
moves tradeably after these alerts (speed is not an edge until measured).

After every alert the script automatically clock-ins the event (see
news_clockin.py): SPY's price at alert time, how much it already moved in
the 15 min before the alert (the scalp window -- 10-30c) and since the
prior close, session volume pace vs the 20d average, and a score -- TRADEABLE_CANDIDATE, ALREADY_MOVED,
OFF_HOURS, or UNKNOWN. Forward 30/60-min SPY outcomes are backfilled by
later runs, completing the study log. Fail-open: clock-in never breaks
alerting.

Read-only. Prints "ALERT [...]" lines; the cron surfaces those to chat.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

from news_clockin import backfill_outcomes, clock_in

PT = ZoneInfo("America/Los_Angeles")
DATA = os.path.expanduser("~/workspace/trading-bot/results/datasets")
SEEN = os.path.join(DATA, "news_wire_seen.json")

FRESH_MIN = 30  # headline must be newer than this to alert

FEEDS = {
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "guardian_world": "https://www.theguardian.com/world/rss",
    "npr": "https://feeds.npr.org/1001/rss.xml",
    "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "investing": "https://www.investing.com/rss/news_25.rss",
}
UA = {"User-Agent": "Mozilla/5.0 (research archive; contact: local)"}

# Tier 1: genuine escalation -- keep tight, these page the user
TIER1 = re.compile(
    r"airstrike|missile strike|missile attack|bombing|invasion|act of war|"
    r"declares war|strike on iran|strikes iran|attack on iran|hormuz.{0,20}"
    r"(clos|block|shut)|drone attack|embassy (attack|storm)|troops deploy|"
    r"nuclear (weapon|test|strike)|sanctions? on iran|"
    r"explosives.{0,30}(found|seized|near)|foiled.{0,20}(attack|plot)|"
    r"bomb plot",
    re.I,
)
# Tier 2: policy shocks that reprice SPY -- require an action verb to cut noise
TIER2 = re.compile(
    r"trump.{0,40}(announces|imposes|raises|threatens|signs|orders|fires|"
    r"removes|vows).{0,40}(tariff|sanction|powell|fed)|"
    r"(fed|federal reserve).{0,40}(emergency|surprise|unscheduled)|"
    r"circuit breaker|trading halt",
    re.I,
)
# Tier 3: de-escalation / off-ramp -- the fade-the-risk-premium moment.
# Checked BEFORE Tier 1 so "lifts sanctions on Iran" doesn't misfire as
# escalation, and a denied/rejected off-ramp flips back to GEOPOLITICAL
# (a denial of talks reprices risk the same way an escalation does).
TIER3_NEG = re.compile(
    r"(denies|denied|rejects?|rejected|dismisses?).{0,40}"
    r"(deal talks|ceasefire|peace|truce|negotiat|roadmap|plan|proposal|offer)",
    re.I,
)
TIER3 = re.compile(
    r"ceasefire|truce|de-?escalat|"
    r"roadmap.{0,30}(end|peace)|"
    r"(reopen|open).{0,30}(strait of hormuz|hormuz)|"
    r"(hormuz|strait).{0,20}reopen|"
    r"(agrees?|open to|offers?|pitches|proposes?|announces?|reaches?|"
    r"signs?|revive).{0,40}(peace|ceasefire|truce|deal|talks|negotiat)|"
    r"(peace|ceasefire).{0,40}(deal|talks|agreed|signed|reached)|"
    r"(lift|eases?|relieve).{0,20}sanctions?|sanctions?.{0,20}(relief|lift|ease)",
    re.I,
)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def load_seen() -> dict:
    try:
        with open(SEEN) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_seen(seen: dict) -> None:
    with open(SEEN, "w") as f:
        json.dump(dict(list(seen.items())[-500:]), f)


def entry_age_min(pub: str | None) -> float | None:
    if not pub:
        return None
    try:
        dt = parsedate_to_datetime(pub.strip())
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60


def spy_volume_ctx() -> str:
    """Today's SPY volume vs 20-day average, for the 'given the volume' read."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="1mo", interval="1d",
                          progress=False, auto_adjust=True)
        if spy is None or len(spy) < 21:
            return ""
        vol = spy["Volume"].dropna()
        today_v = float(vol.iloc[-1])
        avg_v = float(vol.iloc[-21:-1].mean())
        px = float(spy["Close"].iloc[-1])
        prev = float(spy["Close"].iloc[-2])
        chg = (px / prev - 1) * 100
        return (f"SPY ${px:,.2f} ({chg:+.2f}%), "
                f"vol {today_v / avg_v:.1f}x 20d avg")
    except Exception:
        return ""


def main() -> None:
    now = datetime.now(PT)
    ts = now.strftime("%Y-%m-%dT%H:%M")
    seen = load_seen()
    alerts: list[tuple[str, str, str, float]] = []  # tier, source, title, age

    try:
        for source, url in FEEDS.items():
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=20) as r:
                    raw = r.read()
                root = ET.fromstring(raw)
                for item in root.iter("item")[:25]:
                    t_el = item.find("title")
                    title = (t_el.text or "").strip() if t_el is not None else ""
                    if not title:
                        continue
                    key = norm(title)
                    if key in seen:
                        continue
                    p_el = item.find("pubDate")
                    age = entry_age_min(p_el.text if p_el is not None else None)
                    if age is None or age > FRESH_MIN or age < 0:
                        continue
                    tier = None
                    if TIER3_NEG.search(title):
                        tier = "GEOPOLITICAL"  # denied/rejected off-ramp
                    elif TIER3.search(title):
                        tier = "OFF-RAMP"
                    elif TIER1.search(title):
                        tier = "GEOPOLITICAL"
                    elif TIER2.search(title):
                        tier = "POLICY SHOCK"
                    if tier:
                        alerts.append((tier, source, title, age))
                        seen[key] = ts
            except Exception:
                continue
            time.sleep(0.5)
    finally:
        save_seen(seen)

    # Backfill forward SPY outcomes for older alerts (cheap, fail-open).
    try:
        backfill_outcomes()
    except Exception:
        pass

    if not alerts:
        print(f"[{ts} PT] wire: no fresh tradeable headlines")
        return

    ctx = spy_volume_ctx()
    now_pt = datetime.now(PT)
    for tier, source, title, age in alerts:
        print(f"ALERT [{tier}] ({age:.0f}m old) {source}: {title}")
        try:
            # clock_in() also appends the full record to the study log
            rec = clock_in(now_pt, tier, source, title, age)
            move = rec["pre_15m_pct"]
            move_s = f"{move:+.2f}%" if move != "" else "n/a"
            vol = rec["vol_pace_x"]
            vol_s = f"{vol}x" if vol != "" else "n/a"
            nov = rec["novelty_p"]
            nov_s = f"{nov:.2f}" if nov != "" else "n/a"
            print(f"CLOCK-IN [{rec['score']}] SPY ${rec['spy_px_alert']} "
                  f"(pre-alert 15m {move_s}, vol pace {vol_s}, novelty {nov_s}): "
                  f"{rec['reason']}")
        except Exception as e:
            print(f"CLOCK-IN [UNKNOWN] clock-in failed ({e}); alert logged anyway")
    if ctx:
        print(f"CTX: {ctx}")


if __name__ == "__main__":
    main()
