"""News-impulse configuration. Everything env-overridable; disabled by default."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _get_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if raw:
        return [s.strip().upper() for s in raw.split(",") if s.strip()]
    return default


DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "META",
    "AMZN", "GOOGL", "AMD", "JPM", "NFLX",
]

# SEC CIKs for 8-K watch (10-digit, zero-padded). Add more as needed.
DEFAULT_CIKS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "META": "0001326801",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "AMD": "0000002488",
    "JPM": "0000019617",
    "NFLX": "0001065280",
}


@dataclass
class NewsConfig:
    enabled: bool = field(default_factory=lambda: _get("NEWS_IMPULSE_ENABLED", "false").lower() == "true")
    poll_seconds: int = field(default_factory=lambda: _get_int("NEWS_POLL_SECONDS", 45))
    watchlist: List[str] = field(default_factory=lambda: _get_list("NEWS_WATCHLIST", DEFAULT_WATCHLIST))
    ciks: dict = field(default_factory=lambda: dict(DEFAULT_CIKS))

    # event detection
    velocity_bucket_min: int = field(default_factory=lambda: _get_int("NEWS_VELOCITY_BUCKET_MIN", 5))
    velocity_baseline_min: int = field(default_factory=lambda: _get_int("NEWS_VELOCITY_BASELINE_MIN", 120))
    velocity_z: float = field(default_factory=lambda: _get_float("NEWS_VELOCITY_Z", 3.0))

    # verification gate
    verify_window_min: int = field(default_factory=lambda: _get_int("NEWS_VERIFY_WINDOW_MIN", 15))
    verify_min_score: float = field(default_factory=lambda: _get_float("NEWS_VERIFY_MIN_SCORE", 2.0))

    # impulse confirmation
    impulse_mult: float = field(default_factory=lambda: _get_float("NEWS_IMPULSE_MULT", 2.0))
    impulse_lookback_min: int = field(default_factory=lambda: _get_int("NEWS_IMPULSE_LOOKBACK_MIN", 5))
    max_wait_min: int = field(default_factory=lambda: _get_int("NEWS_MAX_WAIT_MIN", 10))
    hold_min: int = field(default_factory=lambda: _get_int("NEWS_HOLD_MIN", 10))

    # risk
    risk_per_event_pct: float = field(default_factory=lambda: _get_float("NEWS_RISK_PER_EVENT_PCT", 0.5))
    max_concurrent_events: int = field(default_factory=lambda: _get_int("NEWS_MAX_CONCURRENT_EVENTS", 3))
    max_daily_loss_pct: float = field(default_factory=lambda: _get_float("NEWS_MAX_DAILY_LOSS_PCT", 2.0))
    max_spread_bps: float = field(default_factory=lambda: _get_float("NEWS_MAX_SPREAD_BPS", 5.0))

    # logging
    signals_log: str = field(default_factory=lambda: _get("NEWS_SIGNALS_LOG", "results/news_impulse_signals.csv"))
    events_log: str = field(default_factory=lambda: _get("NEWS_EVENTS_LOG", "results/news_impulse_events.csv"))
