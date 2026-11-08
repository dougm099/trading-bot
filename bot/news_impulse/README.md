# news-impulse engine

Fast event-reaction trading on public news. Paper-only; emits signals to CSV,
never places orders. Refuses to run with `LIVE_TRADING_ENABLED=true`.

## Pipeline

```
collectors/  ->  events.py  ->  verify.py  ->  impulse.py  ->  risk.py  ->  run_paper.py
 (ingest)       (detect)       (verify)        (confirm)       (size/halt)   (log signals)
```

- **Ingest** (`collectors/`): Arctic Shift Reddit API (primary social source —
  full post bodies across 5 subs, no key), 4chan /biz/ catalog (raw meme
  channel, no auth), ApeWisdom WSB attention rankings (rank-jump spikes only),
  Reddit /new/.rss feeds, Stocktwits (public endpoint dead as of 2026-09-20,
  degrades silently), X v2 recent search (needs `X_BEARER_TOKEN`, secondary),
  EDGAR 8-K Atom feeds (working, no key). RSS headlines already flow via
  `scripts/collect_headlines.py`.
- **Detect** (`events.py`): velocity spike (posts/min z-score over trailing
  baseline) OR keyword taxonomy hit (FDA / MERGER / LAWSUIT / CEO / GUIDANCE /
  BREACH / REGULATORY / EARNINGS). Detects *events*, not sentiment — our own
  sentiment thread died; polarity doesn't predict.
- **Verify** (`verify.py`): weighted multi-source gate. Wire sources (EDGAR,
  CNBC/Reuters/Yahoo/MarketWatch RSS) weight 2.0; social 1.0 per distinct
  handle. A lone tweet never trades.
- **Confirm** (`impulse.py`): after verification, enter only if price moves
  ≥ `impulse_mult` × recent per-minute realized vol within `max_wait_min`,
  in the move's direction. Exits: `hold_min` time stop, 1×-impulse trailing
  stop, 0.5×-impulse hard stop.
- **Risk** (`risk.py`): 0.5% risk/event, ≤3 concurrent, 2% daily-loss halt.
- **Run** (`run_paper.py`): poll loop. Detection-only until a `PriceFeed` is
  attached; even then, signals are only logged.

## Status (2026-09-20)

- Scaffold complete, self-tests pass: `python3 -m bot.news_impulse.selftest`
- Source audit: EDGAR ✓, RSS ✓, Stocktwits ✗ (endpoint killed), X ○ (needs token),
  minute bars ✗ (Stooq JS-walled; path is Alpaca paper once keys are vaulted)
- `NEWS_IMPULSE_ENABLED` defaults false. Promotion bar: 100+ forward paper
  events, costs incl. spread, same DSR/skeptic treatment as the horse race.

## Config

All in `config.py`, env-overridable (`NEWS_*`). Key knobs: `NEWS_WATCHLIST`,
`NEWS_VELOCITY_Z=3.0`, `NEWS_VERIFY_MIN_SCORE=2.0`, `NEWS_IMPULSE_MULT=2.0`,
`NEWS_HOLD_MIN=10`, `NEWS_RISK_PER_EVENT_PCT=0.5`.

## Jev semantic verification filter (`jev_verify.py`)

Complements the multi-source gate: `jev_event_check(symbol, headline, body, category)`
asks TypeSafe's Jev model whether a post describes a real, specific corporate
event vs hype/rumor/recycled news, returning `is_real_event` (probability),
`event_type`, and confidence. Fail-open: returns `None` when the judge CLI is
unavailable, so the source gate still decides. Suggested promotion rule: require
both `gate.is_verified(ev)` AND `is_real_event >= 0.6`. Like everything here,
detection and logging only — social posts alone never trigger trades.
