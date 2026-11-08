#!/usr/bin/env python3
"""Offline A/B: Almgren-Chriss vs TWAP execution on historical bars.

For a grid of (horizon, parent-size) cells, builds AC and TWAP schedules from
trailing bars and works each against *realized* future bars via
bot.execution.simulate_realized_shortfall: the timing component comes from
real price paths; only the temporary-impact concession is modeled
(square-root law calibration). Reports mean/std of realized implementation
shortfall per mode, plus model-based expected temp cost and timing risk.

This is an honest *model-assisted* comparison, not a measurement of real
impact. The true A/B is forward paper: run the bot with --execution ac vs
--execution twap and compare results/execution_fills.csv IS per mode.

Usage:
    python scripts/ab_execution.py --symbol BTCUSDT --bars 4000
    python scripts/ab_execution.py --symbol ETHUSDT --horizons 3600,14400 --sizes-pct-adv 0.5,2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.data.binance import fetch_klines
from bot.execution import build_schedule, simulate_realized_shortfall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--bars", type=int, default=4000)
    ap.add_argument("--horizons", default="3600,14400,43200",
                    help="comma-separated parent-order horizons in seconds")
    ap.add_argument("--sizes-pct-adv", default="0.1,0.5,2.0",
                    help="comma-separated parent sizes as %% of 24h ADV notional")
    ap.add_argument("--n-parents", type=int, default=48,
                    help="parent orders sampled across history (alternate buy/sell)")
    ap.add_argument("--slices", type=int, default=10)
    ap.add_argument("--spread-bps", type=float, default=2.0)
    ap.add_argument("--out", default="results/ab_execution.csv")
    args = ap.parse_args()

    print(f"fetching {args.bars} {args.interval} bars for {args.symbol} ...")
    bars = fetch_klines(args.symbol, args.interval, total=args.bars)
    print(f"got {len(bars)} bars: {bars.index.min()} -> {bars.index.max()}")

    horizons = [float(h) for h in args.horizons.split(",")]
    sizes = [float(s) for s in args.sizes_pct_adv.split(",")]
    step = max(1, len(bars) // (args.n_parents + 2))

    rows = []
    for h in horizons:
        for spct in sizes:
            for k in range(args.n_parents):
                i = len(bars) - 2 - k * step
                if i < 200:
                    break
                hist = bars.iloc[:i]
                start_ts = bars.index[i]
                # 24h ADV notional from trailing bars
                adv = float((hist["close"] * hist["volume"]).tail(24).mean() * 24)
                notional = adv * spct / 100.0
                px = float(bars["close"].iloc[i])
                qty = notional / px
                side = "buy" if k % 2 == 0 else "sell"
                rec = {"horizon_s": h, "size_pct_adv": spct, "side": side,
                       "start": str(start_ts), "notional": round(notional, 2)}
                for mode in ("twap", "ac"):
                    sched = build_schedule(args.symbol, side, qty, px, hist,
                                           mode=mode, horizon_s=h,
                                           n_slices=args.slices)
                    sim = simulate_realized_shortfall(sched, bars, start_ts,
                                                      spread_bps=args.spread_bps)
                    rec[f"{mode}_is_bps"] = round(sim["is_bps"], 3)
                    rec[f"{mode}_timing_bps"] = round(sim["timing_bps"], 3)
                    rec[f"{mode}_impact_bps"] = round(sim["impact_bps"], 3)
                    rec[f"{mode}_exp_temp_bps"] = round(
                        1e4 * sched.exp_temp_cost_usd / sched.notional, 3)
                    rec[f"{mode}_timing_risk_bps"] = round(sched.timing_risk_bps(), 3)
                    rec[f"{mode}_kappa"] = round(sched.kappa, 6)
                rows.append(rec)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} parent orders x2 modes)")

    print("\n=== A/B summary: realized implementation shortfall (bps, lower=better) ===")
    hdr = (f"{'horizon':>8} {'size%ADV':>9} {'n':>3} | "
           f"{'TWAP mean':>9} {'TWAP std':>8} | {'AC mean':>9} {'AC std':>8} | "
           f"{'AC-TWAP':>8}")
    print(hdr)
    for (h, s), g in df.groupby(["horizon_s", "size_pct_adv"]):
        d = g["ac_is_bps"] - g["twap_is_bps"]
        print(f"{h:>8.0f} {s:>9.2f} {len(g):>3} | "
              f"{g['twap_is_bps'].mean():>9.2f} {g['twap_is_bps'].std():>8.2f} | "
              f"{g['ac_is_bps'].mean():>9.2f} {g['ac_is_bps'].std():>8.2f} | "
              f"{d.mean():>+8.2f}")
    print("\n=== Model analytics: expected temp cost / timing risk (bps) ===")
    for (h, s), g in df.groupby(["horizon_s", "size_pct_adv"]):
        print(f"h={h:.0f}s size={s}%ADV: "
              f"TWAP temp={g['twap_exp_temp_bps'].mean():.2f} risk={g['twap_timing_risk_bps'].mean():.2f} | "
              f"AC temp={g['ac_exp_temp_bps'].mean():.2f} risk={g['ac_timing_risk_bps'].mean():.2f} "
              f"(kappa~{g['ac_kappa'].mean():.4f})")
    print("\nRead: AC front-loads (higher expected temp cost, lower timing risk). "
          "A negative AC-TWAP realized mean favors AC on cost; compare std too — "
          "AC's job is cost-per-unit-risk, not always lower mean cost.")


if __name__ == "__main__":
    main()
