"""Weekly paper-portfolio rundown for the long-term ML strategy.

Reads the Alpaca paper account, compares current positions against the
latest rebalance ledger, and prints a chat-ready performance summary.
Read-only: never places orders.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dotenv(path: str = os.path.expanduser("~/workspace/trading-bot/.env")) -> None:
    """Load KEY=VALUE lines into os.environ (user-owned file; never logged)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


load_dotenv()

from bot.brokers.alpaca import AlpacaBroker
from bot.config import Config


def main() -> int:
    cfg = Config()
    if "paper-api.alpaca.markets" not in cfg.alpaca_base_url:
        print("ABORT: broker URL is not the paper sandbox:", cfg.alpaca_base_url)
        return 3
    if not cfg.alpaca_key or not cfg.alpaca_secret:
        print("No Alpaca paper keys configured.")
        return 2

    broker = AlpacaBroker(cfg)
    acct = broker.get_account()
    positions = broker.get_positions()

    ledgers = sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "results", "paper_alpaca", "rebalance_*.json")))
    entry = {}
    asof = "?"
    if ledgers:
        rec = json.load(open(ledgers[-1]))
        asof = rec.get("asof", "?")
        for f in rec.get("fills", []):
            entry[f["symbol"]] = f["fill_price"]

    print(f"PAPER PORTFOLIO — entry {asof}")
    print(f"equity ${acct.equity:,.0f} | cash ${acct.cash:,.0f} | "
          f"{len(positions)} positions")
    print()

    rows = []
    for p in positions:
        ep = entry.get(p.symbol)
        cur = p.current_price if hasattr(p, "current_price") else None
        if ep and cur:
            ret = (cur / ep - 1) * 100
            rows.append((ret, p.symbol, p.qty, ep, cur))
    rows.sort(reverse=True)

    if rows:
        print("winners:")
        for ret, sym, qty, ep, cur in rows[:5]:
            print(f"  {sym:6s} {ret:+6.1f}%  ({qty:g} sh @ ${ep:,.2f} -> ${cur:,.2f})")
        print("losers:")
        for ret, sym, qty, ep, cur in rows[-5:]:
            print(f"  {sym:6s} {ret:+6.1f}%  ({qty:g} sh @ ${ep:,.2f} -> ${cur:,.2f})")

    total_invested = sum(entry.get(p.symbol, 0) * abs(p.qty) for p in positions)
    if total_invested:
        cur_val = sum((p.current_price if hasattr(p, "current_price") else 0)
                      * abs(p.qty) for p in positions)
        print(f"\nportfolio return since entry: {(cur_val / total_invested - 1) * 100:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
