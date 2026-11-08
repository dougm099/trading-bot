"""Test T1 — session-conditional reversal in BTC/ETH (JRFM 2026, 19(9):692).

PRE-REGISTERED SPEC (written before looking at results):
  Sessions defined by US equity hours on CALENDAR days (crypto is 24/7;
  sessions are clock-defined, not equity-calendar-defined):
    - "day"   = bars opening 09:00-15:00 America/New_York  (~09:00-16:00 ET)
    - "night" = bars opening 16:00-08:00 ET next day       (~16:00-09:00 ET)
  Hourly bars; a session is kept only if all its bars are present
  (7 day bars, 17 night bars; DST-transition days are skipped).
  Session return = ln(close of last bar / open of first bar).
  At each session open, position = -sign(previous SAME-session return)
  (i.e. same type, one cycle back — known at open, no lookahead).
  Hold one session, then re-set at the next session open.
  ETH variant: night position = +1 always (standing overnight long);
               day position = -sign(prev day return), as BTC.
  Costs: 10 bps per side. At each session boundary,
         cost = 0.001 * |pos_new - pos_old| (full flip = 20 bps).
  Regimes reported separately: 2016-2020 vs 2021-2025 (+2026 partial).
  Session-boundary sensitivity: day window shifted -2h (07:00-14:00 ET)
  and +2h (11:00-18:00 ET); night = complement.

Skeptic checks (pre-registered):
  S1. Is it just overnight drift? Compare vs buy-and-hold and vs a
      naive "long nights / flat days" benchmark.
  S2. Does it survive costs? Report gross AND net of 10 bps/side.
  S3. Boundary sensitivity: if the edge evaporates at +/-2h, the signal
      is a calendar artifact -> KILL.

Usage: python3 scripts/test_session_reversal.py [--data long|short]
Reads results/cache/crypto_{BTC,ETH}_{long,}.pkl (long = default).
Writes results/test_session_reversal.md + results/session_reversal_summary.json.
Paper math only. No orders, no keys.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

CACHE = os.path.expanduser("~/workspace/trading-bot/results/cache")
OUT_MD = os.path.expanduser("~/workspace/trading-bot/results/test_session_reversal.md")
OUT_JSON = os.path.expanduser("~/workspace/trading-bot/results/session_reversal_summary.json")
COST = 0.001  # 10 bps per side
SESSIONS_PER_YEAR = 730.0


def load(symbol: str, which: str) -> pd.DataFrame:
    name = f"crypto_{symbol}_{'long' if which == 'long' else ''}.pkl".replace("_.pkl", ".pkl")
    # short files are named crypto_BTC.pkl (no suffix)
    path = os.path.join(CACHE, name if which == "long" else f"crypto_{symbol}.pkl")
    df = pd.read_pickle(path)
    assert df.index.tz is not None, "expected tz-aware index"
    return df


def build_sessions(df: pd.DataFrame, day_start: int = 9, day_end: int = 16):
    """Return DataFrame of sessions: date, type, open_px, close_px, ret.
    Fully vectorized (no per-date Python loop)."""
    et = df.index.tz_convert("America/New_York")
    hours = et.hour.to_numpy()
    edates = pd.Index(et.date)
    n_day = day_end - day_start
    n_night = 24 - n_day
    rows = []
    # day sessions: bars with open hour in [day_start, day_end), grouped by date
    dm = (hours >= day_start) & (hours < day_end)
    if dm.any():
        g = df[dm].groupby(edates[dm])
        ok = g.size()
        ok = ok[ok == n_day].index
        if len(ok):
            o = g["open"].first().loc[ok]
            c = g["close"].last().loc[ok]
            for day in ok:
                rows.append((day, "day", o.loc[day], c.loc[day]))
    # night sessions: hour >= day_end belongs to night starting that date;
    # hour < day_start belongs to the night that started the previous date
    nm = (hours >= day_end) | (hours < day_start)
    if nm.any():
        night_date = np.where(hours[nm] >= day_end, edates[nm],
                              edates[nm] - pd.Timedelta(days=1))
        g = df[nm].groupby(night_date)
        ok = g.size()
        ok = ok[ok == n_night].index
        if len(ok):
            o = g["open"].first().loc[ok]
            c = g["close"].last().loc[ok]
            for day in ok:
                rows.append((day, "night", o.loc[day], c.loc[day]))
    s = pd.DataFrame(rows, columns=["date", "type", "open", "close"])
    s["ret"] = np.log(s["close"] / s["open"])
    s = s.sort_values(["date", "type"], ascending=[True, True]).reset_index(drop=True)
    # order within a date: day (09:00) comes before night (16:00)
    return s


def run_strategy(sess: pd.DataFrame, eth_overnight_long: bool = False) -> pd.DataFrame:
    """Attach positions, gross/net session P&L. No lookahead: position for
    session k uses the same-type session return from one cycle back."""
    s = sess.copy()
    s["prev_same_ret"] = s.groupby("type")["ret"].shift(1)
    s["pos"] = -np.sign(s["prev_same_ret"])
    if eth_overnight_long:
        s.loc[s["type"] == "night", "pos"] = 1.0
    s = s.dropna(subset=["prev_same_ret"]).reset_index(drop=True)
    s["prev_pos"] = s["pos"].shift(1).fillna(0.0)
    s["gross"] = s["pos"] * s["ret"]
    s["cost"] = COST * (s["pos"] - s["prev_pos"]).abs()
    s["net"] = s["gross"] - s["cost"]
    return s


def stats(s: pd.DataFrame) -> dict:
    r = s["net"].to_numpy()
    n = len(r)
    mean = r.mean()
    sd = r.std(ddof=1)
    sharpe = (mean / sd * math.sqrt(SESSIONS_PER_YEAR)) if sd > 0 else 0.0
    cum = np.cumsum(r)
    dd = (np.maximum.accumulate(cum) - cum).max()
    tot = math.exp(cum[-1]) - 1 if n else 0.0
    n_trades = int((s["pos"] != s["prev_pos"]).sum())
    turnover = float((s["pos"] - s["prev_pos"]).abs().sum())  # notional units traded
    # gross fade payoff in bp (paper replication diagnostic)
    fade_bp = float(s["gross"].mean() * 1e4)
    tstat = float(mean / (sd / math.sqrt(n))) if sd > 0 and n > 1 else 0.0
    return {
        "sessions": n,
        "total_return_net": round(tot, 4),
        "ann_sharpe_net": round(sharpe, 2),
        "max_drawdown": round(float(dd), 4),
        "gross_fade_payoff_bp": round(fade_bp, 1),
        "tstat_net": round(tstat, 2),
        "n_trades": n_trades,
        "turnover_notional": round(turnover, 1),
        "avg_cost_bp_per_session": round(float(s["cost"].mean() * 1e4), 1),
    }


def regime_stats(s: pd.DataFrame) -> dict:
    out = {}
    s = s.copy()
    s["year"] = pd.to_datetime(s["date"]).dt.year
    for name, mask in [
        ("2016-2020", s["year"].between(2016, 2020)),
        ("2021-2025", s["year"].between(2021, 2025)),
        ("2026+", s["year"] >= 2026),
    ]:
        sub = s[mask]
        out[name] = stats(sub) if len(sub) else {"sessions": 0}
    return out


def benchmark(s: pd.DataFrame, mode: str) -> dict:
    b = s.copy()
    if mode == "buy_hold":
        b["pos"] = 1.0
    elif mode == "overnight_only":
        b["pos"] = np.where(b["type"] == "night", 1.0, 0.0)
    else:
        raise ValueError(mode)
    b["prev_pos"] = b["pos"].shift(1).fillna(0.0)
    b["gross"] = b["pos"] * b["ret"]
    b["cost"] = COST * (b["pos"] - b["prev_pos"]).abs()
    b["net"] = b["gross"] - b["cost"]
    return stats(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="long", choices=["long", "short"])
    args = ap.parse_args()

    all_results = {}
    for symbol in ["BTC", "ETH"]:
        try:
            df = load(symbol, args.data)
        except FileNotFoundError:
            print(f"{symbol}: data file missing, skipping", flush=True)
            continue
        all_results[symbol] = {"data_range": f"{df.index[0]} -> {df.index[-1]}", "bars": len(df)}
        eth_long = symbol == "ETH"
        for label, ds, de in [("base_9_16", 9, 16), ("sens_m2h_7_14", 7, 14), ("sens_p2h_11_18", 11, 18)]:
            sess = build_sessions(df, day_start=ds, day_end=de)
            strat = run_strategy(sess, eth_overnight_long=eth_long)
            res = {
                "sessions_kept": len(sess),
                "overall": stats(strat),
                "regimes": regime_stats(strat),
                "bench_buy_hold": benchmark(strat, "buy_hold"),
                "bench_overnight_only": benchmark(strat, "overnight_only"),
            }
            # gross fade payoff per session type (paper contrast replication)
            for stype in ["day", "night"]:
                sub = strat[strat["type"] == stype]
                v = float(sub["gross"].mean() * 1e4) if len(sub) else 0.0
                res[f"fade_payoff_bp_{stype}"] = round(v, 1)
            all_results[symbol][label] = res
            print(f"{symbol} {label}: sessions={len(strat)} "
                  f"net_sharpe={res['overall']['ann_sharpe_net']} "
                  f"fade_day={res['fade_payoff_bp_day']}bp fade_night={res['fade_payoff_bp_night']}bp",
                  flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    write_md(all_results)
    print(f"Wrote {OUT_MD} and {OUT_JSON}")


def write_md(R: dict) -> None:
    L = []
    L.append("# T1 — Session-conditional reversal in BTC/ETH")
    L.append("")
    L.append("**Test of:** JRFM 2026 19(9):692 — fade the prior same-session return around US equity hours.")
    L.append("**Pre-registered spec:** sessions by US equity hours on calendar days "
             "(day = 09:00–16:00 ET, night = 16:00–09:00 ET); position at each session open = "
             "−sign(previous same-session return); hold one session; flip. "
             "ETH variant: standing +1 overnight long. Costs 10 bps/side on |Δpos|. "
             "Spec written before any results were viewed (see scripts/test_session_reversal.py).")
    L.append("")
    for symbol in ["BTC", "ETH"]:
        if symbol not in R:
            continue
        r = R[symbol]
        L.append(f"## {symbol} (data: {r['data_range']}, {r['bars']} hourly bars)")
        L.append("")
        base = r["base_9_16"]
        o = base["overall"]
        L.append(f"**Base session definition (09:00–16:00 ET), full sample:** "
                 f"{o['sessions']} sessions, net Sharpe {o['ann_sharpe_net']}, "
                 f"total net return {o['total_return_net']:.1%}, max DD {o['max_drawdown']:.1%}, "
                 f"{o['n_trades']} trades, turnover {o['turnover_notional']:.0f} notional units, "
                 f"avg cost {o['avg_cost_bp_per_session']} bp/session. "
                 f"Gross fade payoff: day {base['fade_payoff_bp_day']} bp, night {base['fade_payoff_bp_night']} bp "
                 f"(paper claimed BTC 27.7 / 23.9 bp; ETH day 42.8 bp, overnight long 24.2 bp).")
        L.append("")
        L.append("| regime | sessions | net Sharpe | total net | max DD | fade day bp | fade night bp |")
        L.append("|---|---|---|---|---|---|---|")
        for name, rs in base["regimes"].items():
            if rs.get("sessions", 0):
                L.append(f"| {name} | {rs['sessions']} | {rs['ann_sharpe_net']} | "
                         f"{rs['total_return_net']:.1%} | {rs['max_drawdown']:.1%} | "
                         f"{base['fade_payoff_bp_day']} | {base['fade_payoff_bp_night']} |")
            else:
                L.append(f"| {name} | 0 | – | – | – | – | – |")
        bh = base["bench_buy_hold"]
        on = base["bench_overnight_only"]
        L.append("")
        L.append(f"**Skeptic benchmarks (same sample, same costs):** buy-and-hold net Sharpe {bh['ann_sharpe_net']}, "
                 f"total {bh['total_return_net']:.1%}; long-nights-only net Sharpe {on['ann_sharpe_net']}, "
                 f"total {on['total_return_net']:.1%}.")
        L.append("")
        for lab, title in [("sens_m2h_7_14", "−2h (07:00–14:00 ET)"), ("sens_p2h_11_18", "+2h (11:00–18:00 ET)")]:
            so = r[lab]["overall"]
            L.append(f"**Boundary sensitivity {title}:** net Sharpe {so['ann_sharpe_net']}, "
                     f"total {so['total_return_net']:.1%}, fade day {r[lab]['fade_payoff_bp_day']} bp / "
                     f"night {r[lab]['fade_payoff_bp_night']} bp.")
        L.append("")
    L.append("## Verdict logic (pre-registered)")
    L.append("- PROMOTE only if: net Sharpe > 0.5 in BOTH 2016–2020 and 2021–2025, "
             "survives ±2h boundary shifts (same sign, Sharpe > 0), and beats both skeptic benchmarks.")
    L.append("- KILL if: any regime Sharpe ≤ 0 net of costs, boundary shift flips the sign, "
             "or it fails to beat buy-and-hold / overnight-only.")
    L.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
