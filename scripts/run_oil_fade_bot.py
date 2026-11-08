#!/usr/bin/env python3
"""Oil-fade bot entrypoint.

Default is DRY RUN (signals logged, no orders). --live-paper places real
orders against the Alpaca PAPER sandbox only. There is no live-trading path:
Config.assert_paper_safe() refuses paper-api lookalikes and the runner refuses
LIVE_TRADING_ENABLED=true.

  --once        single poll iteration (for schedulers), then exit
  --live-paper  actually submit paper orders (default: dry run)
  --poll-secs   loop interval (default 60)
"""
import argparse
import logging
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from bot.config import Config
from bot.oil_fade.runner import OilFadePaperRunner
from bot.oil_fade.strategy import OilFadeConfig
from bot.brokers.alpaca import AlpacaBroker
from bot.risk import RiskManager

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ (user-owned file; never logged)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--live-paper", action="store_true")
    ap.add_argument("--poll-secs", type=int, default=60)
    a = ap.parse_args()

    load_dotenv(BASE / ".env")  # Alpaca paper keys live here (chmod 600)
    app_cfg = Config()
    app_cfg.assert_paper_safe()
    if app_cfg.live_trading_enabled:
        raise SystemExit("refusing: LIVE_TRADING_ENABLED is set")

    strat_cfg = OilFadeConfig()  # frozen D2
    broker = AlpacaBroker(app_cfg)
    acct = broker.get_account()
    risk = RiskManager(starting_equity=acct.equity,
                       risk_per_trade_pct=strat_cfg.risk_per_trade_pct,
                       # 1% risk / 0.79% stop distance ~= 133% notional short;
                       # cap at 150% so the risk sizing (not the cap) binds.
                       max_position_pct=150.0,
                       max_daily_loss_pct=strat_cfg.max_daily_loss_pct,
                       max_open_positions=1,
                       stop_loss_pct=strat_cfg.stop_pct)

    state = BASE / "results" / "oil_fade_bot_state.json"
    ledger = BASE / "results" / "datasets" / "oil_fade_paper.csv"
    runner = OilFadePaperRunner(strat_cfg, broker, risk, state, ledger,
                                dry_run=not a.live_paper, poll_secs=a.poll_secs)
    if a.live_paper:
        print("LIVE-PAPER MODE: real orders against the Alpaca paper sandbox.")
    else:
        print("DRY RUN: signals logged, no orders placed.")
    if a.once:
        print("once ->", runner.run_once())
    else:
        runner.run_forever()


if __name__ == "__main__":
    main()
