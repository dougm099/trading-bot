"""Monthly runner: long-horizon ML strategy -> Alpaca paper trading.

Pipeline: fresh yfinance bars -> build_today_features -> 42d HGB bundle
(models/longterm_ml_bundle_42d.joblib) -> top-25 equal weight ->
Alpaca paper orders.

This is REAL-WORLD testing: live market data, real paper fills, fake money.

SAFETY (non-negotiable):
- Hard-pinned to the Alpaca PAPER sandbox. Aborts if Config points at live.
- Dry-run by default: prints picks + planned orders, touches nothing.
- --execute submits market day-orders. Long-only; sells non-targets first.
- Without API keys --execute degrades to a clean no-op report (exit 0).

Usage:
  python scripts/run_longterm_alpaca.py                 # dry-run, $100k assumed
  python scripts/run_longterm_alpaca.py --execute       # submit to paper
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf

from bot.config import Config
from bot.longterm_features import build_today_features
from bot.models import Order
from bot.strategies.longterm_ml import load_bundle, score_universe, target_weights

DATA = os.path.expanduser("~/workspace/trading-bot/results/datasets/longterm")
LEDGER = os.path.expanduser("~/workspace/trading-bot/results/paper_alpaca")
DOTENV = os.path.expanduser("~/workspace/trading-bot/.env")
EXTRA = ["SPY", "^VIX", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
         "XLY", "XLU", "XLRE", "XLB", "XLC"]
BUNDLE = os.path.expanduser("~/workspace/trading-bot/models/longterm_ml_bundle_42d.joblib")
DUST = 1.0  # ignore adjustments under $1


def load_dotenv(path: str = DOTENV) -> None:
    """Load KEY=VALUE lines into os.environ (user-owned file; never logged)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def download(syms: list, batch: int = 100) -> tuple:
    """Batched download (single giant yfinance batches hang); returns
    closes/highs/volumes DataFrames."""
    parts = []
    for k in range(0, len(syms), batch):
        chunk = syms[k:k + batch]
        print(f"  batch {k // batch + 1}/{(len(syms) + batch - 1) // batch} "
              f"({len(chunk)} syms) ...", flush=True)
        d = yf.download(chunk, start="2019-01-01", auto_adjust=True,
                        threads=True, progress=False, timeout=60)
        parts.append(d)
        if d.empty or "Close" not in d:
            raise RuntimeError(f"empty download for batch starting at {chunk[0]}")
    data = pd.concat(parts, axis=1)
    closes = data["Close"].astype(float)
    highs = data["High"].astype(float)
    volumes = data["Volume"].astype(float)
    return closes, highs, volumes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="submit orders to Alpaca paper (default: dry-run)")
    ap.add_argument("--cash", type=float, default=100_000.0,
                    help="assumed equity for dry-run sizing")
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()
    load_dotenv()  # user-owned ~/.env with Alpaca paper keys (optional)

    uni = pd.read_csv(f"{DATA}/universe.csv")["ticker"].tolist()
    sectors = pd.read_csv(f"{DATA}/sector_etf.csv").set_index("ticker")["etf"].to_dict()
    syms = uni + [e for e in EXTRA if e not in uni]
    print(f"downloading {len(syms)} symbols ...", flush=True)
    closes, highs, volumes = download(syms)
    asof = closes.index[-1].date()
    print(f"data as of {asof}, {len(closes)} bars")

    feats = build_today_features(closes, highs, volumes, sectors)
    print(f"scored {len(feats)} tickers")
    bundle = load_bundle(BUNDLE)
    scores = score_universe(feats, bundle)
    picks = scores.head(args.top_n).index.tolist()
    print("top picks:", ", ".join(f"{t}({s:.2f})" for t, s in scores.head(10).items()))

    if not args.execute:
        tgt = target_weights(picks)
        print(f"\nDRY-RUN: equal-weight top-{args.top_n} of ${args.cash:,.0f} "
              f"= ${args.cash / args.top_n:,.2f}/name. No orders sent.")
        print("picks:", ", ".join(picks))
        return 0

    # ---- live-paper path ----
    cfg = Config()
    if "paper-api.alpaca.markets" not in cfg.alpaca_base_url:
        print("ABORT: broker URL is not the paper sandbox:", cfg.alpaca_base_url)
        return 3
    if not cfg.alpaca_key or not cfg.alpaca_secret:
        print(f"DRY-RUN (no ALPACA_API_KEY configured): picks as of {asof}: "
              + ", ".join(picks))
        print("Add paper API keys via Secure Vault to enable order submission.")
        return 0

    from bot.brokers.alpaca import AlpacaBroker
    broker = AlpacaBroker(cfg)
    acct = broker.get_account()
    equity = acct.equity
    print(f"paper account equity ${equity:,.2f}, cash ${acct.cash:,.2f}")
    positions = {p.symbol: p.qty for p in broker.get_positions()}
    tgt_val = equity / args.top_n
    fills, planned = [], []

    # 1. liquidate anything not in the target set (incl. stray shorts)
    for sym, qty in positions.items():
        if sym not in picks and abs(qty) > 0:
            side = "sell" if qty > 0 else "buy"
            planned.append((sym, side, abs(qty), "liquidate non-target"))

    # 2. align targets to equal weight
    for sym in picks:
        last = broker.get_last_price(sym)
        cur_val = positions.get(sym, 0.0) * last
        delta = tgt_val - cur_val
        if abs(delta) >= DUST and last > 0:
            qty = round(abs(delta) / last, 6)
            planned.append((sym, "buy" if delta > 0 else "sell", qty,
                            f"target ${tgt_val:,.0f}"))

    print(f"\nsubmitting {len(planned)} orders to ALPACA PAPER:")
    for sym, side, qty, why in planned:
        print(f"  {side:4s} {qty:>12} {sym:6s}  ({why})")
        fill = broker.place_order(Order(symbol=sym, side=side, qty=qty,
                                        order_type="market", time_in_force="day",
                                        client_id=f"ltml-{asof}-{sym}"))
        fills.append({"symbol": sym, "side": side, "qty": qty,
                      "fill_price": fill.price, "order_id": fill.order_id,
                      "status": fill.status,
                      "ts": fill.timestamp.isoformat()})

    os.makedirs(LEDGER, exist_ok=True)
    rec = {"asof": str(asof), "equity": equity, "picks": picks,
           "scores": {t: round(float(s), 4) for t, s in scores.head(args.top_n).items()},
           "fills": fills,
           "ts": datetime.now(timezone.utc).isoformat()}
    path = f"{LEDGER}/rebalance_{asof}.json"
    json.dump(rec, open(path, "w"), indent=2)
    print(f"\nledger -> {path} ({len(fills)} fills)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
