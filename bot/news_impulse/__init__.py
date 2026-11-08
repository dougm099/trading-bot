"""News-impulse engine: fast event-reaction trading on public news sources.

Pipeline: ingest (Stocktwits / X / RSS / EDGAR) -> event detection (velocity
spike or keyword taxonomy hit) -> verification gate (multi-source) -> impulse
confirmation (trade the market's own first reaction, not the text sentiment)
-> risk-managed paper signals with hard time stops.

PAPER ONLY. This package never places orders. run_paper.py logs signals to
CSV; promotion to any broker requires explicit user authorization and a
forward paper record of 100+ events.
"""
