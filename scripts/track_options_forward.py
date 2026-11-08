"""Forward paper tracking: daily snapshot of ATM straddle pricing (SPY, QQQ)
plus vol indices. Builds the history that free data doesn't provide.
Idempotent: appends one row per symbol per day; regenerates the settled
(realized vs implied) file each run. No orders, no keys -- read-only.
"""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf
from datetime import date, datetime

LOG = os.path.expanduser("~/workspace/trading-bot/results/forward_options_log.csv")
SETTLED = os.path.expanduser("~/workspace/trading-bot/results/forward_options_settled.csv")
SYMBOLS = ["SPY", "QQQ"]

def idx_close(t: str) -> float | None:
    try:
        h = yf.Ticker(t).history(period="2d", auto_adjust=True)
        return float(h["Close"].iloc[-1]) if len(h) else None
    except Exception:
        return None

def snapshot_symbol(sym: str) -> dict | None:
    try:
        t = yf.Ticker(sym)
        und = idx_close(sym)
        if und is None:
            return None
        exps = t.options
        if not exps:
            return None
        today = date.today()
        # nearest expiry with DTE >= 14, closest to 30 DTE
        cand = []
        for e in exps:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if dte >= 14:
                cand.append((abs(dte - 30), dte, e))
        if not cand:
            return None
        _, dte, exp = min(cand)
        ch = t.option_chain(exp)
        calls, puts = ch.calls, ch.puts
        if calls.empty or puts.empty:
            return None
        ci = (calls["strike"] - und).abs().idxmin()
        pi = (puts["strike"] - und).abs().idxmin()
        c, p = calls.loc[ci], puts.loc[pi]
        straddle_mid = float(c["bid"] + c["ask"]) / 2 + float(p["bid"] + p["ask"]) / 2
        return {"date": today.isoformat(), "symbol": sym, "expiry": exp, "dte": dte,
                "underlying": round(und, 2), "strike": float(c["strike"]),
                "call_bid": float(c["bid"]), "call_ask": float(c["ask"]),
                "call_iv": float(c.get("impliedVolatility", float("nan"))),
                "put_bid": float(p["bid"]), "put_ask": float(p["ask"]),
                "put_iv": float(p.get("impliedVolatility", float("nan"))),
                "straddle_mid": round(straddle_mid, 2),
                "straddle_bid": round(float(c["bid"]) + float(p["bid"]), 2),
                "straddle_ask": round(float(c["ask"]) + float(p["ask"]), 2),
                "implied_move_pct": round(100 * straddle_mid / und, 3)}
    except Exception as e:
        print(f"  {sym}: snapshot failed: {str(e)[:100]}")
        return None

def settle():
    """For logged rows whose expiry passed: realized abs move vs implied."""
    if not os.path.exists(LOG):
        return
    log = pd.read_csv(LOG)
    if log.empty:
        return
    rows = []
    px_cache = {}
    for _, r in log.iterrows():
        sym, log_date, expiry = r["symbol"], r["date"], r["expiry"]
        if expiry <= date.today().isoformat():
            if sym not in px_cache:
                px_cache[sym] = yf.Ticker(sym).history(period="max", auto_adjust=True)["Close"]
                px_cache[sym].index = px_cache[sym].index.tz_localize(None)
            px = px_cache[sym]
            try:
                p0 = float(px.loc[:log_date].iloc[-1])
                p1 = float(px.loc[:expiry].iloc[-1])
                realized = 100 * abs(p1 / p0 - 1)
                rows.append({"date": log_date, "symbol": sym, "expiry": expiry,
                             "implied_move_pct": r["implied_move_pct"],
                             "realized_move_pct": round(realized, 3),
                             "overpay_pp": round(r["implied_move_pct"] - realized, 3)})
            except Exception:
                pass
    pd.DataFrame(rows).to_csv(SETTLED, index=False)
    if rows:
        df = pd.DataFrame(rows)
        print(f"  settled {len(df)} straddles; mean overpay {df['overpay_pp'].mean():.2f}pp, "
              f"median {df['overpay_pp'].median():.2f}pp")

def main():
    print(f"[{date.today()}] forward options snapshot")
    new_rows = []
    for sym in SYMBOLS:
        row = snapshot_symbol(sym)
        if row:
            new_rows.append(row)
            print(f"  {sym}: {row['expiry']} {row['dte']}dte strike={row['strike']} "
                  f"straddle_mid={row['straddle_mid']} implied={row['implied_move_pct']}%")
    if os.path.exists(LOG):
        log = pd.read_csv(LOG)
    else:
        log = pd.DataFrame()
    if new_rows:
        new = pd.DataFrame(new_rows)
        dup = False
        if not log.empty:
            dup = ((log["date"] == new_rows[0]["date"]) & (log["symbol"] == new["symbol"].iloc[0])).any()
        if not dup:
            log = pd.concat([log, new], ignore_index=True)
            log.to_csv(LOG, index=False)
            print(f"  appended {len(new)} rows -> {LOG}")
        else:
            print("  today's snapshot already logged")
    # vol index levels
    vols = {t: idx_close(t) for t in ["^VIX", "^VIX3M", "^VVIX", "^SKEW"]}
    print("  vol indices:", {k: (round(v, 2) if v else None) for k, v in vols.items()})
    settle()
    print("done")

if __name__ == "__main__":
    main()
