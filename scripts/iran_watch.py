"""Iran-war market watch: geopolitical tension gauge + NYSE/financial-sector impact.

Runs hourly alongside the hawk snapshots (geopolitics doesn't keep market
hours). Reads the news archive for Iran-related coverage, pulls the latest
daily bars for oil/energy/broad market/financials, and scores tension 0-3.

Outputs:
  results/datasets/iran_watch.csv  ts,iran_mentions_24h,escalation_6h,
      uso_day,xle_day,spy_day,xlf_day,xlf_rel_spy,tnx_day,hyg_day,tension
Prints "ALERT: ..." only when tension >= 2 AND it rose since the last
reading or fresh escalation headlines appeared in the last 2h. The
hourly cron surfaces ALERT lines to chat and stays silent otherwise.
Read-only.
"""
from __future__ import annotations
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

PT = ZoneInfo("America/Los_Angeles")
DATA = os.path.expanduser("~/workspace/trading-bot/results/datasets")
NEWS = os.path.join(DATA, "news_wide.csv")
OUT = os.path.join(DATA, "iran_watch.csv")

IRAN_RE = re.compile(r"iran|tehran|pezeshkian|hormuz|irgc|khamenei", re.I)
ESCALATION_RE = re.compile(
    r"airstrike|missile|sanction|nuclear|military action|troops|strike\b|attack\b|war\b",
    re.I,
)
TICKERS = ["USO", "XLE", "SPY", "XLF", "KBE", "^TNX", "HYG"]


def pct_day(s: pd.Series) -> float:
    # If the latest bar is missing (Yahoo sometimes republishes the daily
    # bar as NaN right after the close), return NaN instead of silently
    # shifting the window back a day. That once misread the prior day's
    # +2.86% USO spike as today's move and phantom-tripped the gauge.
    if len(s) < 2 or pd.isna(s.iloc[-1]) or pd.isna(s.iloc[-2]):
        return float("nan")
    return (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100


def main() -> None:
    now = datetime.now(PT)
    ts = now.strftime("%Y-%m-%dT%H:%M")

    mentions_24h, esc_6h, esc_2h = 0, 0, 0
    latest_headlines: list[str] = []
    if os.path.exists(NEWS):
        news = pd.read_csv(NEWS)
        news["dt"] = pd.to_datetime(news["ts"], format="%Y-%m-%dT%H:%M",
                                    errors="coerce")
        news = news.dropna(subset=["dt"])
        day_ago = now.replace(tzinfo=None) - timedelta(hours=24)
        recent = news[news["dt"] >= pd.Timestamp(day_ago)]
        iran = recent[recent["title"].str.contains(IRAN_RE, na=False)]
        mentions_24h = len(iran)
        esc = iran[iran["title"].str.contains(ESCALATION_RE, na=False)]
        esc_6h = len(esc[esc["dt"] >= pd.Timestamp(now.replace(tzinfo=None)
                                                  - timedelta(hours=6))])
        esc_2h = len(esc[esc["dt"] >= pd.Timestamp(now.replace(tzinfo=None)
                                                  - timedelta(hours=2))])
        latest_headlines = iran.sort_values("dt", ascending=False)["title"]\
            .head(5).tolist()

    closes = yf.download(TICKERS, period="1mo", interval="1d",
                         auto_adjust=True, progress=False)["Close"]
    mv = {t: pct_day(closes[t]) if t in closes else float("nan")
          for t in TICKERS}
    xlf_rel = mv["XLF"] - mv["SPY"] \
        if pd.notna(mv["XLF"]) and pd.notna(mv["SPY"]) else float("nan")

    tension = 0
    if mentions_24h >= 5:
        tension += 1
    if esc_6h >= 1:
        tension += 1
    # war premium building in oil, or financials lagging the market on Iran news
    if (pd.notna(mv["USO"]) and mv["USO"] > 2.0) or \
       (pd.notna(xlf_rel) and xlf_rel < -1.0 and mentions_24h >= 3):
        tension += 1

    row = {"ts": ts, "iran_mentions_24h": mentions_24h, "escalation_6h": esc_6h,
           "uso_day": round(mv["USO"], 2), "xle_day": round(mv["XLE"], 2),
           "spy_day": round(mv["SPY"], 2), "xlf_day": round(mv["XLF"], 2),
           "xlf_rel_spy": round(xlf_rel, 2),
           "tnx_day": round(mv["^TNX"], 2), "hyg_day": round(mv["HYG"], 2),
           "tension": tension}
    prev_tension = None
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        if len(old):
            prev_tension = int(old.iloc[-1]["tension"])
    pd.DataFrame([row]).to_csv(OUT, mode="a",
                               header=not os.path.exists(OUT), index=False)

    print(f"[{ts} PT] iran watch: tension {tension}/3 "
          f"(mentions24h={mentions_24h}, esc6h={esc_6h})")
    print(f"  USO {mv['USO']:+.2f}%  XLE {mv['XLE']:+.2f}%  "
          f"SPY {mv['SPY']:+.2f}%  XLF {mv['XLF']:+.2f}% "
          f"(XLF-SPY {xlf_rel:+.2f}pp)  10Y {mv['^TNX']:+.2f}%  "
          f"HYG {mv['HYG']:+.2f}%")
    for h in latest_headlines[:3]:
        print(f"  - {h[:100]}")
    if tension >= 2 and (prev_tension is None or tension > prev_tension
                         or esc_2h >= 1):
        print(f"ALERT: Iran tension {tension}/3 -- "
              f"{mentions_24h} headlines/24h, {esc_6h} escalation/6h; "
              f"USO {mv['USO']:+.1f}%, XLE {mv['XLE']:+.1f}%, "
              f"SPY {mv['SPY']:+.1f}%, XLF {mv['XLF']:+.1f}% "
              f"(rel {xlf_rel:+.1f}pp), 10Y {mv['^TNX']:+.1f}%")


if __name__ == "__main__":
    main()
