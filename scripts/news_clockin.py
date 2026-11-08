"""Automatic clock-in after every news-wire alert.

For each alert this module snapshots SPY at alert time, measures how much
SPY already moved BEFORE the alert (the "was the trade already gone?"
check), gauges session volume pace vs the 20-day average, and scores the
alert:

  TRADEABLE_CANDIDATE -- fresh headline, market open, no outsized move yet.
  ALREADY_MOVED       -- SPY already made the scalp move (10-30c, ~0.03%) in
                         the 15 min before the alert, or the headline reads as
                         recycled/telegraphed; the trade was before the wire.
                         Do not chase.
  OFF_HOURS           -- market closed; nothing tradeable until the open.
  UNKNOWN             -- data fetch failed; fail-open, alert still logged.

Forward outcomes (SPY return +30/+60 min after the alert) are backfilled by
later wire runs -- no extra cron needed. Everything lands in
news_wire_alerts.csv, the study log.

Scoring is explainable, not a black box: deterministic price checks decide
first, and a Jev novelty judgment ("is this headline genuinely new
information, or recycled/telegraphed?") can only downgrade a candidate to
ALREADY_MOVED, never upgrade noise to a candidate. Thresholds are
documented below and every score carries a human-readable reason; every
input is logged so the study can validate or overturn the thresholds.

Fail-open: any exception -> score UNKNOWN, alert still logged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
DATA = os.path.expanduser("~/workspace/trading-bot/results/datasets")
LOG = os.path.join(DATA, "news_wire_alerts.csv")

OPEN_PT = time(6, 30)
CLOSE_PT = time(13, 0)

# Scoring thresholds (documented, study will validate them).
# Calibrated 2026-09-25 for the user's options-scalp style: the target is a
# 10-30 cent SPY move (~0.013-0.039% at recent prices), quick enter/exit.
# The old 0.50%/60-min trigger was ~13-38x bigger than the trade itself.
PRE_MOVE_THRESH = 0.03   # |%| move in the 15 min before the alert (~20-25c)
                         # -> the scalp already happened, do not chase
NOVELTY_THRESH = 0.30   # Jev novelty p below this -> recycled/telegraphed
# NOTE: pre_open_pct (overnight gap) is logged but no longer forces
# ALREADY_MOVED -- a gap day doesn't remove a 20-cent scalp.
FWD_FILL_MIN = 65       # backfill outcomes once the alert is this old (min)
FWD_GIVEUP_H = 72       # stop retrying unfillable outcomes after this (hours)

JUDGE = os.path.expanduser("~/workspace/skills/typesafe/bin/judge")

COLS = ["ts", "tier", "source", "title", "age_min", "session",
        "spy_px_alert", "px_15m_ago", "px_30m_ago", "px_60m_ago", "prev_close",
        "pre_15m_pct", "pre_30m_pct", "pre_60m_pct", "pre_open_pct",
        "vol_pace_x",
        "novelty_p", "score", "reason",
        "fwd_5m_pct", "fwd_15m_pct", "fwd_30m_pct", "fwd_60m_pct"]


def market_open(ts: datetime) -> bool:
    """Regular session 6:30-13:00 PT on a weekday."""
    return ts.weekday() < 5 and OPEN_PT <= ts.time() < CLOSE_PT


def _fetch(ticker: str, period: str, interval: str):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    import pandas as pd
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.tz_convert(PT)


def _px_at(df, ts: datetime):
    s = df["Close"]
    s = s[s.index <= ts]
    return float(s.iloc[-1]) if len(s) else None


def _daily_ctx():
    """(avg20_vol, prev_close) from daily bars, or (None, None)."""
    d = _fetch("SPY", "1mo", "1d")
    if d is None or len(d) < 22:
        return None, None
    today = datetime.now(PT).date()
    closes = d["Close"].dropna()
    vols = d["Volume"].dropna()
    if len(closes) == 0 or len(vols) < 22:
        return None, None
    last_day = closes.index[-1].date()
    if last_day == today:
        prev_close = float(closes.iloc[-2])
        avg20 = float(vols.iloc[-21:-1].mean())
    else:
        prev_close = float(closes.iloc[-1])
        avg20 = float(vols.iloc[-20:].mean())
    return avg20, prev_close


def novelty_check(headline: str) -> float | None:
    """Jev: is this headline genuinely new information? Fail-open -> None."""
    q = ("Does this headline report genuinely new information about a "
         "market-moving event?")
    ct = ("The headline describes a new, concrete event or action (a strike, "
          "an attack, a signed order, an announced decision) that markets "
          "would not have already priced in")
    cf = ("Recycled news, commentary, opinion, background analysis, a "
          "scheduled speech whose contents were telegraphed in advance, or a "
          "development markets already knew about")
    try:
        out = subprocess.run(
            [JUDGE, "--text", headline, "--noul", q,
             "--criteria-true", ct, "--criteria-false", cf],
            capture_output=True, text=True, timeout=45)
        p = float(json.loads(out.stdout)["judgment"]["noul"])
        return p if 0.0 <= p <= 1.0 else None
    except Exception:
        return None


def score_record(session: str, pre_15m_pct,
                 novelty_p: float | None = None) -> tuple[str, str]:
    """Pure scoring function -- unit-testable without network."""
    if session == "CLOSED":
        return ("OFF_HOURS",
                "market closed; the overnight gap at the open is the trade, "
                "not this headline")
    if pre_15m_pct is not None and abs(pre_15m_pct) >= PRE_MOVE_THRESH:
        return ("ALREADY_MOVED",
                f"SPY already moved {pre_15m_pct:+.2f}% in the 15 min before "
                f"the wire -- the 10-30c scalp already happened. Do not "
                f"chase.")
    if novelty_p is not None and novelty_p < NOVELTY_THRESH:
        return ("ALREADY_MOVED",
                f"reads as recycled/telegraphed rather than new information "
                f"(novelty p={novelty_p:.2f}) -- the market likely priced it "
                f"in before the wire. Do not chase.")
    return ("TRADEABLE_CANDIDATE",
            "fresh headline, market open, no outsized move yet -- watch "
            "price/volume for confirmation before doing anything")


def clock_in(ts_pt: datetime, tier: str, source: str, title: str,
             age_min: float, csv_path: str = LOG) -> dict:
    """Snapshot SPY, score the alert, append to the study log. Fail-open."""
    rec = {c: "" for c in COLS}
    rec.update({
        "ts": ts_pt.strftime("%Y-%m-%dT%H:%M"),
        "tier": tier, "source": source, "title": title,
        "age_min": round(age_min, 1),
        "session": "OPEN" if market_open(ts_pt) else "CLOSED",
    })
    try:
        m1 = _fetch("SPY", "2d", "1m")
        avg20, prev_close = _daily_ctx()
        if m1 is None:
            raise RuntimeError("no intraday bars")
        px = _px_at(m1, ts_pt)
        if px is None:  # pre-open / weekend: use last available close
            px = float(m1["Close"].iloc[-1])
        rec["spy_px_alert"] = round(px, 2)
        p15 = _px_at(m1, ts_pt - timedelta(minutes=15))
        p30 = _px_at(m1, ts_pt - timedelta(minutes=30))
        p60 = _px_at(m1, ts_pt - timedelta(minutes=60))
        if p15:
            rec["px_15m_ago"] = round(p15, 2)
            rec["pre_15m_pct"] = round((px / p15 - 1) * 100, 3)
        if p30:
            rec["px_30m_ago"] = round(p30, 2)
            rec["pre_30m_pct"] = round((px / p30 - 1) * 100, 3)
        if p60:
            rec["px_60m_ago"] = round(p60, 2)
            rec["pre_60m_pct"] = round((px / p60 - 1) * 100, 3)
        if prev_close:
            rec["prev_close"] = round(prev_close, 2)
            rec["pre_open_pct"] = round((px / prev_close - 1) * 100, 3)
        # session volume pace vs 20d average, at the same point in the day
        if rec["session"] == "OPEN" and avg20:
            day = m1[m1.index.date == ts_pt.date()]
            cumvol = float(day.loc[:ts_pt, "Volume"].sum())
            elapsed = max((ts_pt - ts_pt.replace(hour=6, minute=30,
                                                second=0)).total_seconds() / 60, 5)
            expected = avg20 * elapsed / 390
            if expected > 0:
                rec["vol_pace_x"] = round(cumvol / expected, 2)
        novelty = novelty_check(title)
        rec["novelty_p"] = round(novelty, 2) if novelty is not None else ""
        score, reason = score_record(
            rec["session"],
            rec["pre_15m_pct"] if rec["pre_15m_pct"] != "" else None,
            novelty_p=novelty,
        )
    except Exception as e:  # fail-open: never break the wire
        score, reason = "UNKNOWN", f"clock-in data failed ({e}); alert logged anyway"
    rec["score"], rec["reason"] = score, reason
    _append(csv_path, rec)
    return rec


def _append(csv_path: str, rec: dict) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if not exists:
            w.writeheader()
        w.writerow(rec)


def _read(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(csv_path: str, rows: list[dict]) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def backfill_outcomes(csv_path: str = LOG) -> int:
    """Fill fwd_30m/fwd_60m for alerts old enough to have outcomes.

    Called by every wire run; needs no extra cron. Returns rows filled.
    """
    rows = _read(csv_path)
    now = datetime.now(PT)
    todo = []
    for r in rows:
        if r.get("fwd_60m_pct") not in ("", None):
            continue
        try:
            ats = datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M").replace(tzinfo=PT)
        except (ValueError, TypeError):
            continue
        age_min = (now - ats).total_seconds() / 60
        if age_min < FWD_FILL_MIN:
            continue
        if age_min > FWD_GIVEUP_H * 60:
            r["fwd_30m_pct"] = r["fwd_60m_pct"] = "n/a"
            continue
        todo.append((r, ats))
    if not todo:
        # still persist any n/a markings
        _write_all(csv_path, rows)
        return 0
    try:
        m1 = _fetch("SPY", "5d", "1m")
        if m1 is None:
            return 0
    except Exception:
        return 0
    filled = 0
    for r, ats in todo:
        try:
            px_alert = float(r["spy_px_alert"])
        except (ValueError, TypeError):
            continue
        p5 = _px_at(m1, ats + timedelta(minutes=5))
        p15 = _px_at(m1, ats + timedelta(minutes=15))
        p30 = _px_at(m1, ats + timedelta(minutes=30))
        p60 = _px_at(m1, ats + timedelta(minutes=60))
        # only fill when the forward bar is actually after the alert bar
        # (prevents filling with the alert bar itself near the close)
        if p5 and (ats + timedelta(minutes=5)) <= m1.index[-1]:
            r["fwd_5m_pct"] = round((p5 / px_alert - 1) * 100, 3)
        if p15 and (ats + timedelta(minutes=15)) <= m1.index[-1]:
            r["fwd_15m_pct"] = round((p15 / px_alert - 1) * 100, 3)
        if p30 and (ats + timedelta(minutes=30)) <= m1.index[-1]:
            r["fwd_30m_pct"] = round((p30 / px_alert - 1) * 100, 3)
        if p60 and (ats + timedelta(minutes=60)) <= m1.index[-1]:
            r["fwd_60m_pct"] = round((p60 / px_alert - 1) * 100, 3)
            filled += 1
    _write_all(csv_path, rows)
    return filled


def selftest() -> int:
    """End-to-end check with real data: synthetic alerts to a temp CSV."""
    import tempfile
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "test_clockins.csv")
    fails = []

    # 1. pure scoring, no network (scalp-calibrated: 0.03% ~= 20-25c)
    s, _ = score_record("CLOSED", 0.01)
    if s != "OFF_HOURS":
        fails.append("off-hours scoring")
    s, _ = score_record("OPEN", 0.05)
    if s != "ALREADY_MOVED":
        fails.append("pre-move scoring")
    s, _ = score_record("OPEN", -0.04)
    if s != "ALREADY_MOVED":
        fails.append("pre-move negative scoring")
    s, _ = score_record("OPEN", 0.029)  # just under the threshold
    if s != "TRADEABLE_CANDIDATE":
        fails.append("threshold boundary scoring")
    s, _ = score_record("OPEN", 0.01)
    if s != "TRADEABLE_CANDIDATE":
        fails.append("candidate scoring")
    s, _ = score_record("OPEN", None)
    if s != "TRADEABLE_CANDIDATE":
        fails.append("missing-data scoring")
    s, _ = score_record("OPEN", 0.01, novelty_p=0.15)
    if s != "ALREADY_MOVED":
        fails.append("novelty downgrade scoring")
    s, _ = score_record("OPEN", 0.01, novelty_p=0.60)
    if s != "TRADEABLE_CANDIDATE":
        fails.append("novelty keep scoring")
    s, _ = score_record("OPEN", 0.01, novelty_p=None)
    if s != "TRADEABLE_CANDIDATE":
        fails.append("novelty-unknown scoring")
    s, _ = score_record("OPEN", 0.05, novelty_p=0.90)
    if s != "ALREADY_MOVED":
        fails.append("price beats novelty scoring")

    # 1b. live Jev novelty plumbing (fail-open in prod; hard check here)
    p = novelty_check("Trump signs executive order imposing immediate 25% "
                      "tariffs on semiconductor imports")
    print(f"novelty live probe -> p={p}")
    if p is None or not (0.0 <= p <= 1.0):
        fails.append("novelty_check plumbing")

    # 2. replay yesterday's Iran alert (2026-09-23 08:16 PT) against real bars.
    # Manual clock-in that morning concluded the market had already moved on
    # the overnight gap -- the auto score should agree.
    iran_ts = datetime(2026, 9, 23, 8, 16, tzinfo=PT)
    rec = clock_in(iran_ts, "GEOPOLITICAL", "test", "synthetic iran replay",
                   12.0, csv_path=csv_path)
    print(f"replay 2026-09-23 08:16 PT -> score={rec['score']} "
          f"pre15={rec['pre_15m_pct']}% pre60={rec['pre_60m_pct']}% "
          f"preopen={rec['pre_open_pct']}% "
          f"volpace={rec['vol_pace_x']}x novelty={rec['novelty_p']}")
    print(f"  reason: {rec['reason']}")
    if rec["score"] == "UNKNOWN":
        fails.append("replay returned UNKNOWN (data fetch failed)")

    # 3. backfill: fabricate a row 90 min old, confirm fwd outcomes fill.
    old_ts = datetime.now(PT) - timedelta(minutes=90)
    if market_open(old_ts):
        m1 = _fetch("SPY", "2d", "1m")
        px = _px_at(m1, old_ts) if m1 is not None else None
        if px:
            r = {c: "" for c in COLS}
            r.update({"ts": old_ts.strftime("%Y-%m-%dT%H:%M"), "tier": "GEOPOLITICAL",
                      "source": "test", "title": "synthetic backfill",
                      "age_min": 90.0, "session": "OPEN",
                      "spy_px_alert": round(px, 2), "score": "TRADEABLE_CANDIDATE",
                      "reason": "synthetic"})
            _append(csv_path, r)
            n = backfill_outcomes(csv_path)
            rows = _read(csv_path)
            br = [x for x in rows if x["title"] == "synthetic backfill"][0]
            print(f"backfill filled={n} fwd30={br['fwd_30m_pct']}% "
                  f"fwd60={br['fwd_60m_pct']}%")
            if br["fwd_60m_pct"] in ("", None):
                fails.append("backfill did not fill fwd_60m")
    if fails:
        print("SELFTEST FAIL:", fails)
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args()
    if args.backfill:
        print(f"backfilled {backfill_outcomes()} rows")
    elif args.selftest:
        sys.exit(selftest())
    else:
        ap.print_help()
