#!/usr/bin/env python3
"""Score recent news flow with TypeSafe's Jev model.

Reads the newest headlines from the trading-bot news datasets, asks Jev a
standard three-question set (event risk, tone, dominant narrative), and prints
a JSON risk report. Used by the hawk briefing, the monthly rebalance, and
ad-hoc checks.

Usage:
  python3 scripts/news_risk.py [--days 3] [--max-headlines 30] [--context "..."]
                              [--out path.json]

The --context string tells Jev whose risk is being judged, e.g.:
  "Long-only portfolio concentrated in semiconductor and AI-hardware stocks."

Requires the typesafe skill CLI at ~/workspace/skills/typesafe/bin/judge.
Jev judgments are context, not decisions: they never trigger trades alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import date, timedelta

DATA = os.path.expanduser("~/workspace/trading-bot/results/datasets")
JUDGE = os.path.expanduser("~/workspace/skills/typesafe/bin/judge")

SOURCES = [
    ("headlines.csv", "date"),
    ("news_wide.csv", "date"),
]

DEFAULT_CONTEXT = "General US equity market risk."


def load_headlines(days: int, max_headlines: int) -> list[dict]:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    items: list[dict] = []
    for fname, date_col in SOURCES:
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                d = r.get(date_col, "")
                if d and d >= cutoff and r.get("title"):
                    items.append({
                        "date": d,
                        "source": r.get("source", ""),
                        "title": r["title"].strip(),
                    })
    # dedupe by title, keep newest first
    seen: set[str] = set()
    uniq: list[dict] = []
    for it in sorted(items, key=lambda x: x["date"], reverse=True):
        key = it["title"].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq[:max_headlines]


def build_questions(context: str) -> dict:
    return {
        "event_risk": {
            "type": "noul",
            "instructions": (
                "Judge near-term event risk using the recent news headlines in `headlines`. "
                f"Context: {context} "
                "Answer yes if the headlines contain concrete negative developments "
                "(tariff actions, export controls, geopolitical escalation, hawkish policy "
                "signals, major demand warnings, sector-specific scandals) likely to push "
                "the relevant assets down over the next two weeks."
            ),
            "criteria": {
                "true": "Headlines indicate elevated downside event risk over the next two weeks",
                "false": "No elevated downside event risk; news is neutral, positive, or too vague to move prices",
            },
        },
        "tone": {
            "type": "score",
            "instructions": (
                "Rate the net tone of the recent news headlines in `headlines` "
                f"for the following context: {context}"
            ),
            "criteria": [
                "Clearly negative: adverse policy actions, escalation, or demand warnings dominate",
                "Somewhat negative: hawkish or risk-off signals outweigh the positives",
                "Mixed or neutral: no clear directional impulse",
                "Somewhat positive: constructive policy, demand, or de-escalation signals outweigh negatives",
                "Clearly positive: strong demand or breakthrough headlines dominate",
            ],
        },
        "dominant_narrative": {
            "type": "choice",
            "instructions": "Which single narrative best describes the dominant market-moving theme across `headlines`?",
            "criteria": {
                "trade_tariffs": "Trade and tariff developments dominate",
                "monetary_policy": "Central-bank and interest-rate expectations dominate",
                "ai_tech_demand": "AI/tech demand and capex dominate",
                "geopolitics": "Military or security tensions unrelated to trade dominate",
                "corporate": "Company-specific earnings or corporate news dominate",
                "mixed": "No single theme clearly dominates",
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--max-headlines", type=int, default=30)
    ap.add_argument("--context", default=DEFAULT_CONTEXT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    headlines = load_headlines(args.days, args.max_headlines)
    if not headlines:
        print(json.dumps({"error": "no headlines in window", "days": args.days}, indent=1))
        return 1
    if not os.path.exists(JUDGE):
        print(json.dumps({"error": "typesafe skill CLI not found", "path": JUDGE}, indent=1))
        return 2

    state = {"as_of": date.today().isoformat(), "context": args.context, "headlines": headlines}
    with open("/tmp/news_risk_state.json", "w") as f:
        json.dump(state, f)
    with open("/tmp/news_risk_questions.json", "w") as f:
        json.dump(build_questions(args.context), f)

    proc = subprocess.run(
        [JUDGE, "--state-file", "/tmp/news_risk_state.json",
         "--questions-file", "/tmp/news_risk_questions.json"],
        capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        print(json.dumps({"error": "judge CLI failed", "stderr": proc.stderr[-500:]}, indent=1))
        return 3

    answers = json.loads(proc.stdout)
    report = {
        "as_of": date.today().isoformat(),
        "context": args.context,
        "n_headlines": len(headlines),
        "event_risk": answers.get("event_risk", {}).get("noul"),
        "tone": answers.get("tone", {}).get("score"),
        "tone_confidence": answers.get("tone", {}).get("confidence"),
        "dominant_narrative": answers.get("dominant_narrative", {}).get("choice"),
        "narrative_confidence": answers.get("dominant_narrative", {}).get("confidence"),
    }
    text = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
