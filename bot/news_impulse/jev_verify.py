"""Jev semantic verification filter for news-impulse events.

The multi-source gate in verify.py answers "is this event corroborated?"
This module answers the complementary question: "is this post describing a
real, specific corporate event, or is it hype/rumor/old news?": a fast
semantic judgment via the typesafe skill CLI.

It is an additive filter: call `jev_event_check` on a candidate event and
require `is_real_event >= threshold` before promotion. It never places
orders and never overrides the source gate; both must pass.

Requires the typesafe skill CLI at ~/workspace/skills/typesafe/bin/judge.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

JUDGE = os.path.expanduser("~/workspace/skills/typesafe/bin/judge")


def jev_event_check(symbol: str, headline: str, body: str = "",
                   category: str = "", timeout: float = 60.0) -> Optional[dict]:
    """Score whether a social/wire post describes a concrete corporate event.

    Returns {"is_real_event": p, "confidence": c, "event_type": choice} or
    None if the judge CLI is unavailable/fails (fail-open: log and let the
    source gate decide).
    """
    if not os.path.exists(JUDGE):
        return None
    state = {
        "symbol": symbol,
        "detected_category": category,
        "headline": headline,
        "body": (body or "")[:2000],
    }
    questions = {
        "is_real_event": {
            "type": "noul",
            "instructions": (
                f"Social/wire posts mention ticker {symbol} (detected category: "
                f"{category or 'unknown'}). Judge whether the headline and body "
                "describe a REAL, SPECIFIC, CURRENT corporate event — e.g. an "
                "announced merger, FDA decision, earnings release, CEO change, "
                "guidance update, lawsuit filing, or regulatory action — versus "
                "hype, rumor, price-target chatter, recycled old news, memes, "
                "or vague speculation with no concrete development."
            ),
            "criteria": {
                "true": "Describes a real, specific, current corporate event",
                "false": "Hype, rumor, speculation, recycled news, or no concrete event",
            },
        },
        "event_type": {
            "type": "choice",
            "instructions": f"If the post about {symbol} describes a real event, which type is it?",
            "criteria": {
                "merger_acquisition": "Merger, acquisition, or buyout news",
                "regulatory": "FDA, SEC, or other regulatory decision",
                "earnings_guidance": "Earnings release or guidance change",
                "leadership": "CEO/executive change",
                "legal": "Lawsuit, settlement, or investigation",
                "other_event": "A different concrete corporate event",
                "no_event": "No real event described",
            },
        },
    }
    try:
        proc = subprocess.run(
            [JUDGE, "--state", json.dumps(state),
             "--questions", json.dumps(questions)],
            capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return None
        answers = json.loads(proc.stdout)
        return {
            "is_real_event": answers.get("is_real_event", {}).get("noul"),
            "event_type": answers.get("event_type", {}).get("choice"),
            "event_type_confidence": answers.get("event_type", {}).get("confidence"),
        }
    except Exception:
        return None
