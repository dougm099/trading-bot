"""Event model: raw posts, detected events, velocity detector, keyword taxonomy.

Design note: we detect EVENTS (something happened), not SENTIMENT (is it
good/bad). Our own Stocktwits sentiment thread died: polarity scores don't
predict. The impulse engine trades the market's reaction, so detection only
needs to answer "is this novel and real?"
"""
from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional, Set


@dataclass
class RawPost:
    symbol: str
    source: str            # e.g. 'stocktwits', 'x', 'rss:cnbc', 'edgar'
    source_id: str         # handle / domain / feed name: identity for verification
    text: str
    ts: datetime
    url: str = ""


@dataclass
class NewsEvent:
    symbol: str
    category: str          # taxonomy key, or 'velocity' for pure volume spikes
    first_seen: datetime
    sources: Set[str] = field(default_factory=set)      # distinct source_ids
    source_types: Set[str] = field(default_factory=set) # distinct source kinds
    posts: List[RawPost] = field(default_factory=list)
    headline: str = ""

    def add(self, post: RawPost) -> None:
        self.sources.add(post.source_id)
        self.source_types.add(post.source)
        self.posts.append(post)
        if not self.headline:
            self.headline = post.text[:200]


# Keyword taxonomy: category -> compiled patterns. Direction is deliberately
# NOT assigned here: the impulse engine reads direction from price action.
TAXONOMY: Dict[str, List[str]] = {
    "FDA": [r"\bfda\b.*approv", r"approv.*\bfda\b", r"breakthrough therapy",
            r"phase\s*3.*(met|positive|success)", r"advisory committee.*vot"],
    "MERGER": [r"\bacquir", r"\bmerger\b", r"\bbuyout\b", r"\btakeover\b",
               r"\btender offer\b", r"definitive agreement"],
    "LAWSUIT": [r"\bdoj\b", r"\bsec\b.*(charg|sue|probe)", r"class action",
                r"\blawsuit\b", r"antitrust", r"\bftc\b.*block"],
    "CEO": [r"\bceo\b.*(resign|step down|depart|fired|ousted)",
            r"\bceo\b.*(appoint|named|hire)"],
    "GUIDANCE": [r"\bguidance\b.*(rais|cut|lower|miss|beat)",
                 r"\bforecast\b.*(cut|rais)", r"profit warning"],
    "BREACH": [r"data breach", r"\bhack(ed)?\b", r"ransomware", r"cyber ?attack",
               r"security incident"],
    "REGULATORY": [r"\bban(ned|ning)?\b", r"\bsanction", r"export control",
                   r"\btariff\b", r"regulatory.*(halt|suspend)"],
    "EARNINGS": [r"\bq[1-4]\b.*(beat|miss|surprise)", r"earnings.*(beat|miss)",
                 r"revenue.*(beat|miss).*guidance"],
}

_COMPILED = {cat: [re.compile(p, re.IGNORECASE) for p in pats]
             for cat, pats in TAXONOMY.items()}


def classify(text: str) -> Optional[str]:
    """Return the first matching taxonomy category, else None."""
    for cat, patterns in _COMPILED.items():
        if any(p.search(text) for p in patterns):
            return cat
    return None


class VelocityTracker:
    """Per-symbol post-velocity spike detector.

    Buckets post counts into fixed windows; flags when the current bucket's
    count exceeds the trailing baseline mean by velocity_z standard deviations.
    """

    def __init__(self, bucket_min: int = 5, baseline_min: int = 120, z: float = 3.0):
        self.bucket = timedelta(minutes=bucket_min)
        self.baseline = timedelta(minutes=baseline_min)
        self.z = z
        self._counts: Dict[str, Deque] = defaultdict(deque)  # symbol -> deque[(bucket_start, n)]

    def add(self, post: RawPost) -> bool:
        """Record a post; return True if this post completes a spike bucket."""
        sym = post.symbol
        dq = self._counts[sym]
        bstart = post.ts.replace(second=0, microsecond=0)
        bstart -= timedelta(minutes=bstart.minute % int(self.bucket.total_seconds() // 60))
        if dq and dq[-1][0] == bstart:
            dq[-1][1] += 1
        else:
            dq.append([bstart, 1])
        # prune old buckets
        cutoff = post.ts - self.baseline - self.bucket
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        return self._is_spike(sym)

    def _is_spike(self, sym: str) -> bool:
        dq = self._counts[sym]
        if len(dq) < 4:  # need a baseline before crying wolf
            return False
        hist = [n for _, n in list(dq)[:-1]]
        cur = dq[-1][1]
        mean = sum(hist) / len(hist)
        var = sum((n - mean) ** 2 for n in hist) / len(hist)
        sd = math.sqrt(var) if var > 0 else 1.0
        return cur >= mean + self.z * sd and cur >= max(3, int(mean) + 2)
