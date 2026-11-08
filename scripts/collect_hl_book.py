#!/usr/bin/env python3
"""Hyperliquid L2 book + trades collector (public WebSocket, no key needed).

Data collection for future book-level research (OFI, micro-price,
liquidation studies). This script collects data ONLY. No strategy is built on it here.

What it does:
  - Subscribes to Hyperliquid's free public WS (wss://api.hyperliquid.xyz/ws)
    for `l2Book` (20 levels/side) and `trades` on the given coins.
  - Appends L2 snapshots (default every 5 s) and every trade to per-UTC-day
    CSVs under the outdir. Append-only, resumable: restarting just continues
    appending to today's files.
  - Timestamps: exchange `time` (ms) + local receipt time, both UTC.
  - Reconnects with exponential backoff; never dies on a malformed message.

Schema
------
book_YYYYMMDD.csv:
  recv_ts_ms, exch_ts_ms, coin, mid, spread_bps,
  bid_depth_usd, ask_depth_usd, bids_json, asks_json
  (bids_json/asks_json: compact [[px, sz], ...] for the 20 levels)
trades_YYYYMMDD.csv:
  recv_ts_ms, exch_ts_ms, coin, side, px, sz, notional_usd, tid, tx_hash
  (side: 'B' = taker bought (lifted ask), 'A' = taker sold (hit bid))

Size estimate (rough): snapshots ~2 KB/row -> ~35 MB/day/coin at 5 s cadence;
trades ~120 B/row -> tens of MB/day/coin depending on activity. Daily files
make rotation trivial (archive/delete old days).

RUNNING THIS LONG-LIVED  (flagged, not automated here)
----------------------------------------------------------------
This only produces a useful dataset if it runs for weeks/months. Options:

  nohup:   nohup python scripts/collect_hl_book.py >> results/raw_hl_book/collector.log 2>&1 &
  tmux:    tmux new -s hlbook 'python scripts/collect_hl_book.py'
  cron:    @reboot cd ~/workspace/trading-bot && python scripts/collect_hl_book.py
  systemd: write a unit with Restart=always (best for a server/VM).

A cron job / systemd unit has NOT been created by this script. That is a
deliberate human decision (long-lived processes are durable state).
Check the log at results/raw_hl_book/collector.log after starting.

Usage:
    python scripts/collect_hl_book.py --coins BTC,ETH --snapshot-s 5
    python scripts/collect_hl_book.py --outdir /data/hl_book --coins BTC,ETH,SOL
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import signal
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

try:
    import websockets
except ImportError:
    sys.exit("need `websockets`: pip install websockets")

WS_URL = "wss://api.hyperliquid.xyz/ws"
MAX_LEVELS = 20


def utc_ms() -> int:
    return int(time.time() * 1000)


class DayWriter:
    """Append-only CSV writer with UTC-day rotation."""

    def __init__(self, outdir: Path, prefix: str, header: list[str]):
        self.outdir = outdir
        self.prefix = prefix
        self.header = header
        self._day = ""
        self._fh = None
        self._w = None

    def _path(self, day: str) -> Path:
        return self.outdir / f"{self.prefix}_{day}.csv"

    def write(self, row: list) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        if day != self._day or self._fh is None:
            self._rotate(day)
        self._w.writerow(row)
        self._fh.flush()

    def _rotate(self, day: str) -> None:
        if self._fh:
            self._fh.close()
        p = self._path(day)
        new = not p.exists() or p.stat().st_size == 0
        self._fh = p.open("a", newline="")
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(self.header)
        self._day = day

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def parse_book(data: dict) -> dict | None:
    """Parse an l2Book payload defensively. Returns normalized dict or None."""
    try:
        coin = data.get("coin")
        ts = int(data.get("time", 0))
        levels = data.get("levels") or []
        if not coin or len(levels) < 2:
            return None
        bids = [(float(l["px"]), float(l["sz"])) for l in levels[0][:MAX_LEVELS]]
        asks = [(float(l["px"]), float(l["sz"])) for l in levels[1][:MAX_LEVELS]]
        if not bids or not asks:
            return None
        best_bid, best_ask = bids[0][0], asks[0][0]
        mid = 0.5 * (best_bid + best_ask)
        spread_bps = 1e4 * (best_ask - best_bid) / mid if mid > 0 else 0.0
        return {
            "coin": str(coin), "exch_ts_ms": ts,
            "bids": bids, "asks": asks, "mid": mid, "spread_bps": spread_bps,
            "bid_depth_usd": sum(px * sz for px, sz in bids),
            "ask_depth_usd": sum(px * sz for px, sz in asks),
        }
    except Exception:
        return None


def parse_trade(t: dict) -> dict | None:
    try:
        px, sz = float(t["px"]), float(t["sz"])
        return {
            "coin": str(t.get("coin", "")),
            "exch_ts_ms": int(t.get("time", 0)),
            "side": str(t.get("side", "")),
            "px": px, "sz": sz,
            "notional_usd": px * sz,
            "tid": t.get("tid", ""),
            "tx_hash": t.get("hash", ""),
        }
    except Exception:
        return None


async def run(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("hlbook")
    log.info("collector starting: coins=%s snapshot_s=%s outdir=%s",
             args.coins, args.snapshot_s, outdir)

    books: dict[str, dict] = {}
    trade_buf: deque[list] = deque()
    seen_tids: dict[str, set] = defaultdict(set)

    book_w = DayWriter(outdir, "book",
                       ["recv_ts_ms", "exch_ts_ms", "coin", "mid", "spread_bps",
                        "bid_depth_usd", "ask_depth_usd", "bids_json", "asks_json"])
    trade_w = DayWriter(outdir, "trades",
                        ["recv_ts_ms", "exch_ts_ms", "coin", "side", "px", "sz",
                         "notional_usd", "tid", "tx_hash"])

    stop = asyncio.Event()

    def _sig(*_a):
        stop.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, _sig)
        except NotImplementedError:
            pass

    def flush_trades() -> None:
        while trade_buf:
            trade_w.write(trade_buf.popleft())

    async def snapshot_loop() -> None:
        while not stop.is_set():
            await asyncio.sleep(args.snapshot_s)
            now = utc_ms()
            for coin in args.coins:
                b = books.get(coin)
                if not b:
                    continue
                book_w.write([
                    now, b["exch_ts_ms"], coin,
                    round(b["mid"], 6), round(b["spread_bps"], 4),
                    round(b["bid_depth_usd"], 2), round(b["ask_depth_usd"], 2),
                    json.dumps([[round(px, 6), round(sz, 8)] for px, sz in b["bids"]],
                               separators=(",", ":")),
                    json.dumps([[round(px, 6), round(sz, 8)] for px, sz in b["asks"]],
                               separators=(",", ":")),
                ])
            flush_trades()

    backoff = 1.0
    snap_task = asyncio.create_task(snapshot_loop())
    try:
        while not stop.is_set():
            try:
                async with websockets.connect(WS_URL, max_size=20_000_000,
                                              ping_interval=20, ping_timeout=20) as ws:
                    for coin in args.coins:
                        await ws.send(json.dumps(
                            {"method": "subscribe",
                             "subscription": {"type": "l2Book", "coin": coin}}))
                        await ws.send(json.dumps(
                            {"method": "subscribe",
                             "subscription": {"type": "trades", "coin": coin}}))
                    log.info("subscribed: %s", args.coins)
                    backoff = 1.0
                    async for raw in ws:
                        if stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        ch = msg.get("channel")
                        data = msg.get("data")
                        if ch == "l2Book" and isinstance(data, dict):
                            b = parse_book(data)
                            if b:
                                books[b["coin"]] = b
                        elif ch == "trades" and isinstance(data, list):
                            now = utc_ms()
                            for t in data:
                                tr = parse_trade(t)
                                if not tr or not tr["coin"]:
                                    continue
                                tid = str(tr["tid"])
                                if tid and tid in seen_tids[tr["coin"]]:
                                    continue
                                if tid:
                                    seen_tids[tr["coin"]].add(tid)
                                    if len(seen_tids[tr["coin"]]) > 200_000:
                                        seen_tids[tr["coin"]] = set(
                                            list(seen_tids[tr["coin"]])[-100_000:])
                                trade_buf.append([
                                    now, tr["exch_ts_ms"], tr["coin"], tr["side"],
                                    tr["px"], tr["sz"], round(tr["notional_usd"], 4),
                                    tid, tr["tx_hash"],
                                ])
                            if len(trade_buf) > 5000:
                                flush_trades()
            except Exception as e:
                if stop.is_set():
                    break
                log.warning("ws error (%s); reconnecting in %.0fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
    finally:
        snap_task.cancel()
        flush_trades()
        book_w.close()
        trade_w.close()
        log.info("collector stopped cleanly")


def main() -> None:
    global WS_URL
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", default="BTC,ETH",
                    help="comma-separated Hyperliquid coin names")
    ap.add_argument("--snapshot-s", type=float, default=5.0,
                    help="L2 snapshot cadence in seconds")
    ap.add_argument("--outdir", default="results/raw_hl_book")
    ap.add_argument("--ws-url", default=WS_URL)
    args = ap.parse_args()
    args.coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    WS_URL = args.ws_url

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(outdir / "collector.log"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
