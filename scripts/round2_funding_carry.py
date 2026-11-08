#!/usr/bin/env python3
"""Round 2: H5 Hyperliquid funding-rate carry (delta-neutral).

Structural trade: long spot + short perp, equal notional. P&L per 8h funding
interval = funding payment collected (positive = longs pay shorts, we are short)
+ basis change (premium_entry - premium_exit amortized) - fees.
No prediction involved: this harvests a mechanical payment.

Signal: trailing-10d mean annualized funding. Enter when ann. avg
funding > ENTER_T (default 10%); exit when < EXIT_T (default 5%).
Fees: perp taker 3.5bps + spot taker 5bps per leg per side -> 17bps round trip.
Sensitivity: also run at 2x fees.

Data: Hyperliquid public API fundingHistory (hourly), spot hourly from
Coinbase via bot/data/binance.py. Cached to results/cache/hl_funding_<COIN>.csv.
Regimes: split sample in half (regime by time); also report by BTC 90d trend sign.

No lookahead: entry/exit decisions use trailing funding only.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from bot.data.binance import fetch_klines  # noqa: E402

HL = "https://api.hyperliquid.xyz/info"
CACHE = BASE / "results" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
COINS = ["BTC", "ETH", "SOL"]
ENTER_T = 0.10   # annualized trailing funding to enter
EXIT_T = 0.05    # annualized trailing funding to exit
FEE_RT = 0.0017  # 17bps round trip (perp 3.5bps + spot 5bps, each side)


def fetch_funding(coin, chunk_days=90):
    """Page fundingHistory back as far as the API gives. Returns DataFrame.

    NOTE: Hyperliquid pays funding every 8h; each row = one 8h funding
    interval (timestamps land on 00:00/08:00/16:00 UTC), NOT hourly.
    """
    cache = CACHE / f"hl_funding_{coin}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True,
                         date_format="mixed")
        df.index = pd.to_datetime(df.index, utc=True)
        if len(df) > 100:
            return df
    out, end = [], int(time.time() * 1000)
    while True:
        start = end - chunk_days * 86400 * 1000
        r = requests.post(HL, json={"type": "fundingHistory", "coin": coin,
                                    "startTime": start, "endTime": end}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out = batch + out
        end = batch[0]["time"] - 1
        if len(batch) < 10:
            break
        time.sleep(0.3)
        if len(out) > 30000:
            break
    df = pd.DataFrame(out)
    df["t"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["premium"] = df["premium"].astype(float)
    df = df.set_index("t").sort_index()[["fundingRate", "premium"]]
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(cache)
    return df


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 50 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def backtest(coin, fee_mult=1.0, enter_t=ENTER_T, exit_t=EXIT_T):
    """Backtest in funding-interval space (each row = 8h). Delta-neutral P&L
    per interval in position = funding payment for that interval, plus
    premium capture (entry_prem - exit_prem), minus fees. No spot prices
    needed: direction cancels by construction."""
    fund = fetch_funding(coin).sort_index()
    df = fund.rename(columns={"fundingRate": "fr", "premium": "prem"})
    if len(df) < 200:
        return None
    df["fr_ann"] = df["fr"] * 3 * 365          # 8h intervals -> annualized
    df["trail10d"] = df["fr_ann"].rolling(30).mean()  # 30 intervals = 10 days
    pos, entry_prem, rets = 0, 0.0, []
    trades = 0
    for _, row in df.iterrows():
        if pos == 0 and row["trail10d"] > enter_t:
            pos, entry_prem, trades = 1, row["prem"], trades + 1
            rets.append(row["fr"] - FEE_RT * fee_mult / 2)
        elif pos == 1 and row["trail10d"] < exit_t:
            pos = 0
            rets.append(row["fr"] + (entry_prem - row["prem"])
                        - FEE_RT * fee_mult / 2)
        elif pos == 1:
            rets.append(row["fr"])
        else:
            rets.append(0.0)
    rets = pd.Series(rets, index=df.index)
    bh = df["fr"]  # always-in gross carry per interval
    eq = (1 + rets).cumprod()
    n = len(rets)
    thirds = [0, n // 3, 2 * n // 3, n]
    out = {"coin": coin, "n_int": n,
           "from": str(df.index[0].date()), "to": str(df.index[-1].date()),
           "trades": trades,
           "tot": float(eq.iloc[-1] - 1), "sharpe": ann_sharpe(rets, 3 * 365),
           "maxdd": max_dd(eq),
           "bh_carry_tot": float((1 + bh).cumprod().iloc[-1] - 1),
           "fee_mult": fee_mult, "enter_t": enter_t}
    for i, name in enumerate(["R1", "R2", "R3"]):
        sub = rets.iloc[thirds[i]:thirds[i + 1]]
        e2 = (1 + sub).cumprod()
        out[f"{name}_tot"] = float(e2.iloc[-1] - 1)
        out[f"{name}_sharpe"] = ann_sharpe(sub, 3 * 365)
        out[f"{name}_avgfr"] = float(df["fr_ann"].iloc[thirds[i]:thirds[i + 1]].mean())
    return out


if __name__ == "__main__":
    print("===== H5 Hyperliquid funding-rate carry (delta-neutral) =====")
    print(f"params: enter>{ENTER_T:.0%} ann trailing-7d funding, exit<{EXIT_T:.0%}, "
          f"fees {FEE_RT*1e4:.0f}bps round-trip")
    for coin in COINS:
        try:
            r = backtest(coin)
            if r is None:
                print(f"  {coin}: insufficient data"); continue
            print(f"  {coin} {r['from']}->{r['to']} n={r['n_int']} 8h-intervals trades={r['trades']}")
            print(f"    net tot={r['tot']:+.2%} Sharpe={r['sharpe']:+.2f} maxDD={r['maxdd']:.2%}")
            for R in ["R1", "R2", "R3"]:
                print(f"    | {R} tot={r[R+'_tot']:+.2%} Sh={r[R+'_sharpe']:+.2f} avg_ann_fund={r[R+'_avgfr']:.1%}")
            print(f"    always-in gross carry={r['bh_carry_tot']:+.2%} (before fees)")
            r2 = backtest(coin, fee_mult=2.0)
            print(f"    [2x fees] net tot={r2['tot']:+.2%} Sharpe={r2['sharpe']:+.2f}")
        except Exception as e:
            print(f"  {coin}: ERROR {e}")
    print("DONE")
