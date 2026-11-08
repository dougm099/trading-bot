"""Hawk briefing generator: reads the hawk datasets and prints a markdown
briefing -- regime snapshot, correlation shifts, sector rotation,
news pulse, and auto-flagged anomalies.

Run daily after the close (via cron). The cron worker adds its own
read on top of this quantitative base before delivering to chat.
Read-only.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PT = ZoneInfo("America/Los_Angeles")
DATA = os.path.expanduser("~/workspace/trading-bot/results/datasets")
SNAP = os.path.join(DATA, "hawk_snapshots.csv")
NEWS = os.path.join(DATA, "news_wide.csv")

WATCHLIST = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY",
    "^VIX", "^TNX", "^FVX",
    "GLD", "USO", "UUP", "TLT", "HYG",
    "EFA", "EEM",
    "BTC-USD", "ETH-USD",
]
# user's own stocks (added 2026-09-21) -- shown in their own briefing section
USER_STOCKS = ["ETH-USD", "AAPL", "PLTR", "HIMS", "RCAT"]
WATCHLIST = WATCHLIST + [s for s in USER_STOCKS if s not in WATCHLIST]
REGIME_BOARD = ["SPY", "QQQ", "IWM", "^VIX", "^TNX", "BTC-USD",
                "GLD", "UUP", "TLT", "USO", "EEM"]
PAIRS = [("SPY", "TLT"), ("SPY", "GLD"), ("QQQ", "BTC-USD"),
         ("SPY", "UUP"), ("GLD", "^TNX"), ("^VIX", "SPY"),
         ("XLE", "USO"), ("EEM", "SPY")]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY"]

THEMES = {
    "tariffs/trade": ["tariff", "trade war", "trade deal", "duties", "import tax"],
    "Fed/rates": ["federal reserve", "powell", "rate cut", "rate hike", "interest rate", "fomc"],
    "inflation": ["inflation", "consumer prices", "cpi "],
    "US politics": ["election", "senate", "congress", "white house", "supreme court"],
    "war/conflict": ["war ", "missile", "airstrike", "ceasefire", "sanctions", "troops"],
    "energy": ["opec", "crude", "oil prices", "gasoline"],
    "AI/tech": ["artificial intelligence", "openai", "nvidia", "datacenter"],
    "crypto": ["bitcoin", "ethereum", "crypto"],
}


def rolling_corr(rets: pd.DataFrame, a: str, b: str, win: int) -> float:
    s = rets[[a, b]].dropna()
    if len(s) < win:
        return float("nan")
    return s[a].rolling(win).corr(s[b]).iloc[-1]


def main() -> None:
    today = datetime.now(PT).strftime("%Y-%m-%d")
    print(f"# Hawk briefing — {today}\n")

    closes = yf.download(WATCHLIST, period="6mo", interval="1d",
                         auto_adjust=True, progress=False)["Close"]
    rets = closes.pct_change()

    def last_valid(s: pd.Series, n: int = 1) -> float:
        s = s.dropna()
        return float(s.iloc[-n]) if len(s) >= n else float("nan")

    day, w5, last = {}, {}, {}
    for sym in WATCHLIST:
        if sym not in closes:
            continue
        s = closes[sym]
        last[sym] = last_valid(s, 1)
        prev = last_valid(s, 2)
        w5[sym] = (last[sym] / last_valid(s, 6) - 1) * 100
        day[sym] = (last[sym] / prev - 1) * 100 if prev else float("nan")

    print("## Regime board (day %)")
    for sym in REGIME_BOARD:
        if sym in day:
            lvl = f"{last[sym]:,.2f}"
            extra = f"  10Y {last[sym]:.2f}%" if sym == "^TNX" else ""
            print(f"- {sym:8s} {day[sym]:+6.2f}%  (last {lvl}){extra}")
    print()

    print("## Your stocks")
    for sym in USER_STOCKS:
        if sym in day:
            name = {"ETH-USD": "ETH", "AAPL": "AAPL", "PLTR": "PLTR",
                    "HIMS": "HIMS", "RCAT": "RCAT"}.get(sym, sym)
            print(f"- {name:8s} day {day[sym]:+6.2f}%   5d {w5[sym]:+6.2f}%")
    print()

    print("## Correlation shifts (20d vs 60d)")
    flagged = []
    for a, b in PAIRS:
        if a not in rets or b not in rets:
            continue
        c20 = rolling_corr(rets, a, b, 20)
        c60 = rolling_corr(rets, a, b, 60)
        if pd.isna(c20) or pd.isna(c60):
            continue
        d = c20 - c60
        mark = "  <-- SHIFT" if abs(d) > 0.25 else ""
        if mark:
            flagged.append((a, b, c20, c60, d))
        print(f"- {a}/{b}: 20d {c20:+.2f} vs 60d {c60:+.2f} (Δ {d:+.2f}){mark}")
    print()

    print("## Sector rotation")
    sec = [(s, day[s], w5[s]) for s in SECTORS if s in day]
    sec.sort(key=lambda x: x[1], reverse=True)
    for s, d, w in sec:
        print(f"- {s:5s} day {d:+5.2f}%   5d {w:+6.2f}%")
    print()

    print("## News pulse (today)")
    if os.path.exists(NEWS):
        news = pd.read_csv(NEWS)
        td = news[news["date"] == today]
        if len(td):
            by_cat = td.groupby("category").size().sort_values(ascending=False)
            for cat, n in by_cat.items():
                print(f"- {cat}: {n} headlines")
            titles = " ".join(td["title"].str.lower().fillna(""))
            print("\ntheme hits:")
            for theme, kws in THEMES.items():
                hits = sum(titles.count(k) for k in kws)
                if hits:
                    print(f"- {theme}: {hits}")
        else:
            print("- no headlines captured today yet")
    else:
        print("- news archive not started")
    print()

    print("## Iran watch (geopolitics -> NYSE / financials)")
    IW = os.path.join(DATA, "iran_watch.csv")
    if os.path.exists(IW):
        iw = pd.read_csv(IW)
        hist = iw.tail(7)[["ts", "tension", "iran_mentions_24h",
                           "uso_day", "xle_day", "spy_day",
                           "xlf_day", "xlf_rel_spy"]]
        for _, r in hist.iterrows():
            print(f"- {r['ts']}: tension {int(r['tension'])}/3, "
                  f"USO {r['uso_day']:+.1f}%, XLE {r['xle_day']:+.1f}%, "
                  f"SPY {r['spy_day']:+.1f}%, XLF {r['xlf_day']:+.1f}% "
                  f"(rel {r['xlf_rel_spy']:+.1f}pp), "
                  f"{int(r['iran_mentions_24h'])} headlines/24h")
    else:
        print("- no readings yet")
    print()

    print("## Auto-flags")
    flags = []
    if "^VIX" in day and "SPY" in day and day["^VIX"] > 3 and day["SPY"] > 0.5:
        flags.append("VIX spiking while SPY is green -- fear without price damage; watch for reversal or hedge demand.")
    if "^VIX" in day and day["^VIX"] > 10:
        flags.append(f"VIX +{day['^VIX']:.0f}% -- fear spike; check news pulse for the trigger.")
    for a, b, c20, c60, d in flagged:
        flags.append(f"{a}/{b} correlation regime shift: {c60:+.2f} -> {c20:+.2f}.")
    if "BTC-USD" in day and "QQQ" in day and abs(day["BTC-USD"] - day["QQQ"]) > 4:
        flags.append(f"BTC ({day['BTC-USD']:+.1f}%) decoupling from QQQ ({day['QQQ']:+.1f}%) -- crypto-specific driver.")
    if "GLD" in day and "^TNX" in day and day["GLD"] > 1 and day["^TNX"] > 0:
        flags.append("Gold up while yields rise -- unusual; possible safe-haven bid on non-rate fears.")
    if "UUP" in day and day["UUP"] > 0.5 and day["SPY"] < -0.5:
        flags.append("Dollar up + stocks down -- classic risk-off.")
    if not flags:
        flags.append("No major anomalies vs recent regime.")
    for f in flags:
        print(f"- {f}")


if __name__ == "__main__":
    main()
