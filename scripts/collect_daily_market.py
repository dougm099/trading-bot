"""Daily close-of-market collection: runs every close-based own-dataset
collector in one shot. Intraday (2:49pm PT) collectors (ATM straddles,
earnings straddles, vol surface) run under the forward-options-tracking
cron instead, since option quotes are live then.

Runs:
  1. collect_intraday_bars.py  -- hourly OHLCV backfill+append, 19 symbols
  2. collect_vol_indices.py     -- VIX / VIX3M / VIX6M / VVIX / SKEW closes
  3. collect_headlines.py       -- finance headlines archive for sentiment work

Idempotent per script. Read-only: no orders, no keys.
"""
from __future__ import annotations
import runpy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    for script in ["collect_intraday_bars.py",
                   "collect_vol_indices.py",
                   "collect_headlines.py"]:
        print(f"\n=== {script} ===")
        try:
            runpy.run_path(os.path.join(HERE, script), run_name="__main__")
        except SystemExit:
            pass
        except Exception as e:
            print(f"  ERROR in {script}: {str(e)[:200]}")
    print("\ndaily market collection done")


if __name__ == "__main__":
    main()
