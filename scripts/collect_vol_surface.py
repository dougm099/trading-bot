"""Own-dataset builder: vol-surface snapshots (SPY, QQQ, IWM).

Why: no free source offers historical option prices/IVs. The existing
ATM tracker covers one point on the surface; this adds the wings and
term structure so future research can answer skew, term-structure, and
VRP-timing questions honestly.

Per symbol x expiry bucket (front / ~30d / ~60d), one row per day:
  - ATM straddle mid + ATM call/put IV
  - wing IVs: put struck nearest 95% of spot, call struck nearest 105%
  - 25-delta-ish skew proxy: put95_iv - call105_iv

Runs intraday alongside track_options_forward.py (option quotes are
live then; post-close quotes are stale). Idempotent: one row per
(date, symbol, bucket). Read-only: no orders, no keys.
"""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf
from datetime import date, datetime

OUT = os.path.expanduser("~/workspace/trading-bot/results/datasets/vol_surface.csv")
SYMBOLS = ["SPY", "QQQ", "IWM"]
# (bucket label, target DTE, minimum DTE)
BUCKETS = [("front", 7, 1), ("m30", 30, 14), ("m60", 60, 30)]


def pick_expiry(exps: list[str], target: int, mindte: int, today: date):
    cand = []
    for e in exps:
        dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        if dte >= mindte:
            cand.append((abs(dte - target), dte, e))
    if not cand:
        return None, None
    _, dte, exp = min(cand)
    return exp, dte


def wing_iv(chain: pd.DataFrame, spot: float, side: str, frac: float):
    """IV of the strike nearest frac*spot on the given side's chain."""
    target = spot * frac
    try:
        i = (chain["strike"] - target).abs().idxmin()
        row = chain.loc[i]
        return float(row["strike"]), float(row.get("impliedVolatility", float("nan")))
    except Exception:
        return float("nan"), float("nan")


def snapshot(sym: str, bucket: str, target: int, mindte: int, today: date):
    try:
        t = yf.Ticker(sym)
        und = float(t.history(period="2d", auto_adjust=True)["Close"].iloc[-1])
        exps = t.options
        if not exps:
            return None
        exp, dte = pick_expiry(list(exps), target, mindte, today)
        if not exp:
            return None
        ch = t.option_chain(exp)
        calls, puts = ch.calls, ch.puts
        if calls.empty or puts.empty:
            return None
        ci = (calls["strike"] - und).abs().idxmin()
        pi = (puts["strike"] - und).abs().idxmin()
        c, p = calls.loc[ci], puts.loc[pi]
        straddle_mid = (float(c["bid"]) + float(c["ask"]) +
                        float(p["bid"]) + float(p["ask"])) / 2
        put95_k, put95_iv = wing_iv(puts, und, "put", 0.95)
        call105_k, call105_iv = wing_iv(calls, und, "call", 1.05)
        skew = (put95_iv - call105_iv
                if pd.notna(put95_iv) and pd.notna(call105_iv) else float("nan"))
        return {"date": today.isoformat(), "symbol": sym, "bucket": bucket,
                "expiry": exp, "dte": dte, "underlying": round(und, 2),
                "atm_strike": float(c["strike"]),
                "straddle_mid": round(straddle_mid, 3),
                "call_iv": round(float(c.get("impliedVolatility", float("nan"))), 4),
                "put_iv": round(float(p.get("impliedVolatility", float("nan"))), 4),
                "put95_strike": round(put95_k, 2), "put95_iv": round(put95_iv, 4),
                "call105_strike": round(call105_k, 2),
                "call105_iv": round(call105_iv, 4),
                "skew_put95_call105": round(skew, 4),
                "implied_move_pct": round(100 * straddle_mid / und, 3)}
    except Exception as e:
        print(f"  {sym}/{bucket}: failed: {str(e)[:100]}")
        return None


def main():
    today = date.today()
    print(f"[{today}] vol surface snapshot")
    rows = []
    for sym in SYMBOLS:
        for bucket, target, mindte in BUCKETS:
            r = snapshot(sym, bucket, target, mindte, today)
            if r:
                rows.append(r)
                print(f"  {sym}/{bucket}: {r['expiry']} {r['dte']}dte "
                      f"straddle={r['straddle_mid']} skew={r['skew_put95_call105']}")
    if not rows:
        print("  no snapshots (market holiday or Yahoo outage?)")
        return
    new = pd.DataFrame(rows)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        key = {"date", "symbol", "bucket"}
        have = set(zip(old["date"], old["symbol"], old["bucket"]))
        new = new[~new.apply(lambda r: (r["date"], r["symbol"], r["bucket"]) in have, axis=1)]
        if new.empty:
            print("  today's surface already logged")
            return
        old = pd.concat([old, new], ignore_index=True)
        old.to_csv(OUT, index=False)
    else:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        new.to_csv(OUT, index=False)
    print(f"  +{len(new)} rows -> {OUT}")


if __name__ == "__main__":
    main()
