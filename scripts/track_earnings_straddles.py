"""Forward paper tracking: earnings straddles.
Watches liquid names; when earnings are within 3 days, snapshots the
nearest-expiry ATM straddle (bid/ask/IV). After earnings, settles:
implied move % vs realized earnings-day move %.
Answers over time: do straddle buyers systematically overpay for earnings?
Read-only: no orders, no keys.
"""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta

LOG = os.path.expanduser("~/workspace/trading-bot/results/earnings_straddle_log.csv")
WATCH = ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD", "NFLX", "JPM"]

def next_earnings(sym: str) -> date | None:
    t = yf.Ticker(sym)
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed is not None:
                ed = ed[0] if isinstance(ed, (list, tuple)) else ed
                d = pd.Timestamp(ed).date()
                return d
    except Exception:
        pass
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            fut = ed[ed.index.date >= date.today()]
            if not fut.empty:
                return fut.index[0].date()
    except Exception:
        pass
    return None

def snapshot(sym: str, earn: date) -> dict | None:
    try:
        t = yf.Ticker(sym)
        und = float(t.history(period="2d", auto_adjust=True)["Close"].iloc[-1])
        exps = [e for e in t.options
                if datetime.strptime(e, "%Y-%m-%d").date() >= earn]
        if not exps:
            return None
        exp = min(exps, key=lambda e: datetime.strptime(e, "%Y-%m-%d").date())
        ch = t.option_chain(exp)
        ci = (ch.calls["strike"] - und).abs().idxmin()
        pi = (ch.puts["strike"] - und).abs().idxmin()
        c, p = ch.calls.loc[ci], ch.puts.loc[pi]
        mid = (float(c["bid"]) + float(c["ask"]) + float(p["bid"]) + float(p["ask"])) / 2
        return {"log_date": date.today().isoformat(), "symbol": sym,
                "earnings_date": earn.isoformat(), "expiry": exp,
                "underlying": round(und, 2), "strike": float(c["strike"]),
                "straddle_bid": round(float(c["bid"]) + float(p["bid"]), 2),
                "straddle_ask": round(float(c["ask"]) + float(p["ask"]), 2),
                "straddle_mid": round(mid, 2),
                "call_iv": float(c.get("impliedVolatility", float("nan"))),
                "put_iv": float(p.get("impliedVolatility", float("nan"))),
                "implied_move_pct": round(100 * mid / und, 3),
                "status": "pre", "realized_move_pct": "", "overpay_pp": ""}
    except Exception as e:
        print(f"  {sym}: snapshot failed: {str(e)[:100]}")
        return None

def settle(log: pd.DataFrame) -> pd.DataFrame:
    px_cache = {}
    for i, r in log.iterrows():
        if r["status"] != "pre":
            continue
        earn = datetime.strptime(r["earnings_date"], "%Y-%m-%d").date()
        if earn >= date.today():
            continue
        sym = r["symbol"]
        try:
            if sym not in px_cache:
                h = yf.Ticker(sym).history(period="3mo", auto_adjust=False)
                h.index = h.index.tz_localize(None)
                px_cache[sym] = h
            h = px_cache[sym]
            # realized = |close(earn day) / close(day before) - 1|, earnings-day session
            idx = h.index.date
            pos = list(idx).index(earn)
            prev_close = float(h["Close"].iloc[pos - 1])
            earn_close = float(h["Close"].iloc[pos])
            earn_open = float(h["Open"].iloc[pos])
            realized = 100 * abs(earn_close / prev_close - 1)
            gap = 100 * abs(earn_open / prev_close - 1)
            log.at[i, "realized_move_pct"] = round(realized, 3)
            log.at[i, "gap_pct"] = round(gap, 3)
            log.at[i, "overpay_pp"] = round(float(r["implied_move_pct"]) - realized, 3)
            log.at[i, "status"] = "settled"
            print(f"  settled {sym} {earn}: implied {r['implied_move_pct']}% vs "
                  f"realized {realized:.2f}% (gap {gap:.2f}%)")
        except Exception as e:
            print(f"  {sym}: settle failed: {str(e)[:80]}")
    return log

def read_log():
    try:
        if os.path.exists(LOG) and os.path.getsize(LOG) > 0:
            return pd.read_csv(LOG)
    except Exception:
        pass
    return pd.DataFrame()

def main():
    print(f"[{date.today()}] earnings straddle watch")
    log = read_log()
    for sym in WATCH:
        earn = next_earnings(sym)
        if earn is None:
            print(f"  {sym}: no earnings date found")
            continue
        d = (earn - date.today()).days
        print(f"  {sym}: earnings {earn} (in {d}d)")
        if 0 <= d <= 3:
            already = (not log.empty and ((log["symbol"] == sym) &
                       (log["earnings_date"] == earn.isoformat())).any())
            if already:
                print("    already logged")
                continue
            row = snapshot(sym, earn)
            if row:
                log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
                print(f"    snapshot: straddle_mid={row['straddle_mid']} "
                      f"implied={row['implied_move_pct']}%")
    log = settle(log)
    if "gap_pct" not in log.columns and not log.empty:
        log["gap_pct"] = ""
    log.to_csv(LOG, index=False)
    print(f"  log -> {LOG} ({len(log)} rows)")
    s = log[log["status"] == "settled"] if "status" in log.columns else log.iloc[0:0]
    if len(s):
        print(f"  settled sample n={len(s)}: mean overpay {pd.to_numeric(s['overpay_pp']).mean():.2f}pp")

if __name__ == "__main__":
    main()
