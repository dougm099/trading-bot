"""Thread 2: funding / basis arbitrage extensions.
A) Cross-venue funding gap: long perp where funding most negative, short where
   most positive (delta-neutral), Hyperliquid vs dYdX, full fees + margin cost.
B) Cross-venue cash-and-carry: buy BTC spot (Coinbase), short BTC perp (HL),
   collect funding + basis convergence. No transfer needed (unwind in place).
Bar vs H5: H5 found single-venue HL funding carry below T-bills; the bar here
is risk-free rate, not zero.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

HL = "https://api.hyperliquid.xyz/info"
DYDX = "https://indexer.dydx.trade/v4"
CB = "https://api.exchange.coinbase.com"

# Fees (retail tiers): HL taker 0.035%, dYdX taker 0.05%, Coinbase taker 0.60%
FEE_HL = 0.00035
FEE_DYDX = 0.0005
FEE_CB = 0.0060
TBILL = 0.04  # benchmark


def hl_funding(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    out = []
    s = start_ms
    while s < end_ms:
        r = requests.post(HL, json={"type": "fundingHistory", "coin": coin,
                                    "startTime": s}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        s = batch[-1]["time"] + 1
        if len(batch) < 500:
            break
        time.sleep(0.15)
    df = pd.DataFrame(out)
    df["t"] = pd.to_datetime(df["time"].astype(int), unit="ms", utc=True)
    df["fr"] = df["fundingRate"].astype(float)
    df["premium"] = df["premium"].astype(float)
    return df[["t", "fr", "premium"]].sort_values("t").reset_index(drop=True)


def dydx_funding(ticker: str, hours: int) -> pd.DataFrame:
    out, before = [], None
    while len(out) < hours:
        params = {"limit": 100}
        if before:
            params["effectiveBeforeOrAt"] = before
        r = requests.get(f"{DYDX}/historicalFunding/{ticker}", params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()["historicalFunding"]
        if not batch:
            break
        out.extend(batch)
        before = batch[-1]["effectiveAt"]
        if len(batch) < 100:
            break
        time.sleep(0.15)
    df = pd.DataFrame(out)
    df["t"] = pd.to_datetime(df["effectiveAt"], utc=True)
    df["fr"] = df["rate"].astype(float)
    return df[["t", "fr"]].drop_duplicates("t").sort_values("t").reset_index(drop=True)


def hl_candles(coin: str, start_ms: int, end_ms: int, interval: str = "1h") -> pd.DataFrame:
    out, s = [], start_ms
    while s < end_ms:
        r = requests.post(HL, json={"type": "candle", "coin": coin, "interval": interval,
                                    "startTime": s, "endTime": end_ms}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        s = batch[-1]["T"] + 1
        if len(batch) < 500:
            break
        time.sleep(0.15)
    df = pd.DataFrame(out)
    df["t"] = pd.to_datetime(df["T"].astype(int), unit="ms", utc=True)
    df["close"] = df["c"].astype(float)
    return df[["t", "close"]].sort_values("t").reset_index(drop=True)


def cb_hourly(n: int) -> pd.DataFrame:
    out, end = [], None
    while len(out) < n:
        params = {"granularity": 3600}
        if end:
            params["end"] = end.isoformat()
            params["start"] = (end - pd.Timedelta(hours=300)).isoformat()
        r = requests.get(f"{CB}/products/BTC-USD/candles", params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out = batch + out
        end = pd.to_datetime(batch[-1][0], unit="s", utc=True)
        time.sleep(0.2)
    df = pd.DataFrame(out, columns=["t", "low", "high", "open", "close", "volume"])
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["t", "close"]].drop_duplicates("t").sort_values("t").tail(n).reset_index(drop=True)


def annualized(fr: pd.Series) -> pd.Series:
    return fr * 24 * 365


def main():
    now_ms = int(time.time() * 1000)
    start_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)

    print("Fetching HL funding (BTC)...")
    hl = hl_funding("BTC", start_ms, now_ms)
    print("Fetching dYdX funding (BTC-USD)...")
    dy = dydx_funding("BTC-USD", hours=9000)
    print(f"HL rows={len(hl)} from {hl['t'].min()} | dYdX rows={len(dy)} from {dy['t'].min()}")

    # ---- A) cross-venue funding gap ----
    g = pd.merge_asof(hl.rename(columns={"fr": "hl"}).sort_values("t"),
                      dy.rename(columns={"fr": "dy"}).sort_values("t"),
                      on="t", direction="nearest", tolerance=pd.Timedelta("30min")).dropna()
    g["gap"] = g["hl"] - g["dy"]  # hourly-rate difference
    g["gap_ann"] = annualized(g["gap"].abs())
    print("\n=== A) Cross-venue funding gap (HL vs dYdX, BTC) ===")
    print(g["gap_ann"].describe(percentiles=[0.5, 0.9, 0.99]).to_string())
    print(f"median |gap| annualized: {g['gap_ann'].median():.2%}")
    print(f"99th pct |gap|: {g['gap_ann'].quantile(0.99):.2%}")

    # Strategy: enter when |gap_ann| > 3%, exit when < 1%. Long the cheap-funding venue.
    # Fees per round trip: 2*(FEE_HL + FEE_DYDX) = 0.17% of notional.
    # Margin: assume 10% locked per venue => 20% of notional locked; opportunity cost TBILL.
    fee_rt = 2 * (FEE_HL + FEE_DYDX)
    g["pos"] = 0
    in_pos, entry_t, entry_gap, capital = False, None, 0.0, 1.0
    trades, pnl = 0, 0.0
    for _, row in g.iterrows():
        if not in_pos and abs(row["gap_ann"]) > 0.03:
            in_pos, entry_t, entry_gap = True, row["t"], row["gap"]
            trades += 1
            pnl -= fee_rt  # entry fees on full notional
        elif in_pos:
            # collect hourly gap differential on notional: we are long the venue
            # with the more negative funding, i.e. we receive |gap| per hour
            pnl += abs(row["gap"])
            if abs(row["gap_ann"]) < 0.01:
                in_pos = False
                pnl -= fee_rt  # exit fees
    years = (g["t"].max() - g["t"].min()).total_seconds() / 31536000
    print(f"\nSimulated delta-neutral gap strategy: {trades} round trips, "
          f"net P&L {pnl:.2%} of notional over {years:.2f}y, fees {fee_rt:.2%}/trip")
    if years >= 0.5:
        print(f"annualized net of fees: {pnl/years:.2%}/yr on notional; "
              f"~20% margin locked => ~{pnl/years/0.2:.2%}/yr on locked margin "
              f"(before margin opportunity cost ~{TBILL:.0%}) vs T-bill {TBILL:.0%}")
    else:
        print(f"(sample < 6 months: {pnl:.2%} total, not annualized)")
    g[["t", "hl", "dy", "gap_ann"]].to_csv("results/arb_funding_gap.csv", index=False)

    # ---- B) cash-and-carry: long spot / short HL perp ----
    # No BTC spot market on HL; basis = fundingHistory 'premium' (perp vs mark,
    # hourly, 2024-01 -> now). Spot leg assumed filled at mark (Coinbase ~ mark).
    # Fees per round trip: 2x(spot taker 0.60% + perp taker 0.035%).
    print("\n=== B) Cash-and-carry: long spot / short HL perp (basis = premium) ===")
    fee_cc = 2 * (FEE_CB + FEE_HL)
    print(f"round-trip fee hurdle: {fee_cc:.3%}")
    f = hl.copy()
    f["basis"] = f["premium"]
    print(f["basis"].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).to_string())
    wins = (f["basis"] > fee_cc).mean()
    print(f"hours where basis > fee hurdle: {wins:.3%}")
    # Simulate: enter when premium > 1.5% (above hurdle w/ margin), exit < 0.1%.
    # While in: collect funding (short receives +fr), capture basis convergence.
    in_pos, pnl_cc, trades_cc, in_hours = False, 0.0, 0, 0
    for _, row in f.iterrows():
        if not in_pos and row["basis"] > 0.015:
            in_pos, entry_basis = True, row["basis"]
            trades_cc += 1
            pnl_cc -= fee_cc
        elif in_pos:
            in_hours += 1
            pnl_cc += row["fr"]  # short perp: receive fr when fr>0, pay when fr<0
            if row["basis"] < 0.001:
                pnl_cc += entry_basis - row["basis"]  # convergence captured
                in_pos = False
    if in_pos:  # still open at end: mark at last basis
        pnl_cc += entry_basis - f["basis"].iloc[-1]
    years_cc = (f["t"].max() - f["t"].min()).total_seconds() / 31536000
    ret_cap = pnl_cc / years_cc / 1.1  # 110% capital employed (spot + perp margin)
    print(f"cash-and-carry: {trades_cc} trips, P&L {pnl_cc:.2%} of notional over {years_cc:.2f}y")
    print(f"=> {ret_cap:.2%}/yr on capital vs T-bill {TBILL:.0%}; in-position {in_hours/len(f):.1%} of time")
    f[["t", "basis", "fr"]].to_csv("results/arb_cash_carry.csv", index=False)

    summary = {
        "funding_gap": {
            "hours": len(g),
            "median_abs_gap_ann_pct": round(float(g["gap_ann"].median() * 100), 3),
            "p99_abs_gap_ann_pct": round(float(g["gap_ann"].quantile(0.99) * 100), 3),
            "round_trips": trades,
            "pnl_pct_of_notional": round(pnl * 100, 3),
            "years": round(years, 2),
        },
        "cash_carry": {
            "hours": len(f),
            "fee_hurdle_pct": round(fee_cc * 100, 3),
            "hours_basis_above_hurdle_frac": round(float(wins), 4),
            "round_trips": trades_cc,
            "pnl_pct_of_notional": round(pnl_cc * 100, 3),
            "years": round(years_cc, 2),
        },
    }
    with open("results/arb_funding_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nwrote results/arb_funding_gap.csv, results/arb_cash_carry.csv, results/arb_funding_summary.json")


if __name__ == "__main__":
    main()
