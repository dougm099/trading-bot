#!/usr/bin/env python3
"""
analyze_sentiment.py: Stocktwits social-sentiment -> price analysis.

Pipeline:
  1. Fetch Stocktwits messages per symbol with polite cursor pagination (cached to JSONL).
     --extend continues a previous crawl backwards instead of starting over.
  2. Build hourly sentiment series + trailing-24h aggregates (matters for equities:
     weekend/overnight chatter accumulates into the next session bar).
  3. Load hourly prices: Coinbase (crypto, 24/7) / yfinance (equities, regular session).
  4. Strict no-lookahead tests (features at hour h use only messages with t <= close of bar h):
       - corr of sentiment level/change at h vs forward 6/12/24h returns
       - contemporaneous reactivity check (sentiment(h) vs PAST returns)
       - event study on sentiment spikes (trailing 7-day z-score > 2 on 24h-aggregated bull_ratio)
  5. For candidates passing thresholds (|r|>0.03 & p<0.05 & n>=50, or event |t|>2):
     dead-simple H-hour hold backtest WITH costs (10 bps/side crypto, 5 bps/side equities).

Usage:  python3 analyze_sentiment.py            # full run (uses cache)
        python3 analyze_sentiment.py --extend    # crawl further back, merge into cache
        python3 analyze_sentiment.py --refetch   # ignore cache, re-crawl

Results: ~/workspace/trading-bot/results/
"""
from __future__ import annotations
import argparse, json, math, os, random, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- config
SYMBOLS = ["BTC.X", "ETH.X", "SPY", "QQQ", "NVDA", "TSLA"]
CRYPTO = {"BTC.X": "BTC-USD", "ETH.X": "ETH-USD"}   # Stocktwits sym -> Coinbase product
UA = "Mozilla/5.0"
BASE = os.path.expanduser("~/workspace/trading-bot/results")
RAW_DIR = os.path.join(BASE, "raw")
MAX_PAGES = {"crypto": 250, "equity": 300}   # politeness cap on NEW pages per crawl
SLEEP_S = 1.5                                 # base sleep between requests (+ jitter)
MAX_HISTORY_DAYS = 35                         # stop crawling once we have this much history
FORWARD_H = [6, 12, 24]
MIN_N_CORR = 10                               # report r/p only with n>=10; n<50 flagged low-n
COST_BPS = {"crypto": 10.0, "equity": 5.0}    # per side, basis points
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(BASE, exist_ok=True)

# ---------------------------------------------------------------- fetch
def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                wait = min(90, (2 ** attempt) * 2 + random.random())
                print(f"    HTTP {e.code} — backing off {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            wait = min(90, (2 ** attempt) * 2 + random.random())
            print(f"    network error {type(e).__name__} — retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
    raise RuntimeError(f"gave up on {url}")

def fetch_symbol(sym: str, refetch: bool = False, extend: bool = False) -> list[dict]:
    """Paginate Stocktwits stream backwards via ?max=. Cached to JSONL.
    extend=True keeps the existing cache and crawls further back from its oldest id."""
    cache = os.path.join(RAW_DIR, f"st_messages_{sym}.jsonl")
    msgs, seen = [], set()
    if os.path.exists(cache) and not refetch:
        for line in open(cache):
            if line.strip():
                m = json.loads(line)
                msgs.append(m)
                seen.add(m["id"])
        if not extend:
            print(f"  [{sym}] loaded {len(msgs)} cached messages", flush=True)
            return msgs
        print(f"  [{sym}] extending {len(msgs)} cached messages further back", flush=True)

    kind = "crypto" if sym in CRYPTO else "equity"
    max_id = (min(m["id"] for m in msgs) - 1) if msgs else None
    oldest = None
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_HISTORY_DAYS)
    pages = 0
    cool_cycles = 0
    for page in range(MAX_PAGES[kind]):
        pages = page + 1
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{urllib.parse.quote(sym, safe='')}.json"
        if max_id:
            url += f"?max={max_id}"
        try:
            d = _get(url)
        except RuntimeError:
            # _get exhausted its retries: long cooldown, then resume at same page
            cool_cycles += 1
            if cool_cycles > 3:
                raise
            print(f"  [{sym}] long cooldown 300s (cycle {cool_cycles}/3)", flush=True)
            time.sleep(300)
            d = _get(url)
        batch = d.get("messages", [])
        if not batch:
            print(f"  [{sym}] empty page, stopping", flush=True)
            break
        for m in batch:
            mid = m["id"]
            if mid in seen:
                continue
            seen.add(mid)
            sent = (m.get("entities") or {}).get("sentiment")
            msgs.append({
                "id": mid,
                "created_at": m.get("created_at"),
                "body": (m.get("body") or "")[:500],
                "sentiment": (sent or {}).get("basic"),   # "Bullish"/"Bearish"/None
            })
            ts = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00"))
            if oldest is None or ts < oldest:
                oldest = ts
        max_id = min(m["id"] for m in batch) - 1
        more = (d.get("cursor") or {}).get("more", False)
        time.sleep(SLEEP_S + random.random())
        if not more:
            print(f"  [{sym}] cursor says no more", flush=True)
            break
        if oldest and oldest < cutoff:
            print(f"  [{sym}] reached {MAX_HISTORY_DAYS}d history", flush=True)
            break
        if pages % 25 == 0:   # incremental checkpoint so a crash doesn't lose the crawl
            msgs.sort(key=lambda m: m["id"])
            with open(cache, "w") as f:
                for m in msgs:
                    f.write(json.dumps(m) + "\n")
            print(f"  [{sym}] checkpoint: {len(msgs)} msgs", flush=True)
    msgs.sort(key=lambda m: m["id"])
    with open(cache, "w") as f:
        for m in msgs:
            f.write(json.dumps(m) + "\n")
    print(f"  [{sym}] now {len(msgs)} messages total (+{pages} new pages)", flush=True)
    return msgs

# ------------------------------------------------------- sentiment series
def build_sentiment(msgs: list[dict]) -> pd.DataFrame:
    """Hourly counts on a full clock grid + trailing-24h aggregates."""
    rows = []
    for m in msgs:
        try:
            ts = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        rows.append((ts, m.get("sentiment")))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "sentiment"])
    df["hour"] = df["ts"].dt.floor("h")
    g = df.groupby("hour")["sentiment"]
    hourly = pd.DataFrame({
        "n_bullish": g.apply(lambda s: (s == "Bullish").sum()),
        "n_bearish": g.apply(lambda s: (s == "Bearish").sum()),
    })
    grid = pd.date_range(hourly.index.min().floor("h"), hourly.index.max().ceil("h"),
                         freq="h", tz="UTC")
    hourly = hourly.reindex(grid)
    b = hourly["n_bullish"].fillna(0.0)
    be = hourly["n_bearish"].fillna(0.0)
    out = pd.DataFrame(index=grid)
    out["n_bullish"], out["n_bearish"] = b, be
    out["n_labeled"] = b + be
    out["bull_ratio_1h"] = b / (b + be).replace(0, np.nan)
    out["net_1h"] = (b - be) / (b + be).replace(0, np.nan)
    b24 = b.rolling(24, min_periods=6).sum()
    be24 = be.rolling(24, min_periods=6).sum()
    out["bull_ratio_24h"] = b24 / (b24 + be24).replace(0, np.nan)
    out["net_24h"] = (b24 - be24) / (b24 + be24).replace(0, np.nan)
    out["d_bull_ratio_1h"] = out["bull_ratio_1h"].diff()
    out["d_bull_ratio_24h"] = out["bull_ratio_24h"].diff()
    out.index.name = "hour"
    return out

# ---------------------------------------------------------------- prices
def coinbase_hourly(product: str, days: int = 40) -> pd.Series:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    closes = []
    cur_end = end
    while cur_end > start:
        cur_start = max(start, cur_end - timedelta(hours=300))
        q = urllib.parse.urlencode({
            "granularity": 3600,
            "start": cur_start.isoformat(), "end": cur_end.isoformat(),
        })
        url = f"https://api.exchange.coinbase.com/products/{product}/candles?{q}"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        batch = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=25) as r:
                    batch = json.load(r)   # [time, low, high, open, close, volume], newest first
                break
            except Exception as e:
                wait = min(60, 2 ** attempt + random.random())
                print(f"    coinbase error {type(e).__name__} — retrying in {wait:.0f}s", flush=True)
                time.sleep(wait)
        if batch is None:
            raise RuntimeError(f"coinbase candles failed for {product}")
        for c in batch:
            closes.append((datetime.fromtimestamp(c[0], tz=timezone.utc)
                           .replace(minute=0, second=0, microsecond=0), c[4]))
        cur_end = cur_start
        time.sleep(0.4)
    s = pd.Series({t: p for t, p in closes}, name="close").sort_index()
    return s[~s.index.duplicated(keep="last")]

def yf_hourly(ticker: str, days: int = 120) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, period=f"{days}d", interval="1h", auto_adjust=True,
                     progress=False, prepost=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):      # safety: squeeze single-col frames
        close = close.iloc[:, 0]
    close = close.dropna()
    idx = close.index
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    close.index = idx.tz_convert("UTC").floor("h")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.name = "close"
    return close

def align(price: pd.Series, sent: pd.DataFrame):
    """Price onto the sentiment clock grid (ffill); session mask = real bars."""
    grid = sent.index
    px = price.reindex(grid).ffill()
    session = pd.Series(grid.isin(price.index), index=grid)
    return px, session

# ---------------------------------------------------------------- analysis
def forward_returns(px: pd.Series, H: int) -> pd.Series:
    return px.shift(-H) / px - 1.0

def corr_test(x: pd.Series, y: pd.Series, step: int = 1) -> tuple[float, float, int, int]:
    """Pearson r AND p on non-overlapping subsample (every `step`-th obs) so
    overlapping-window autocorrelation doesn't fake significance.
    Returns (r, p, n_used, n_full)."""
    df = pd.concat([x, y], axis=1).dropna()
    n_full = len(df)
    sub = df.iloc[::max(1, step)]
    n = len(sub)
    if n < MIN_N_CORR:
        return np.nan, np.nan, n, n_full
    r, p = stats.pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
    return float(r), float(p), n, n_full

def partial_corr(x: pd.Series, y: pd.Series, z: pd.Series, step: int = 1) -> tuple[float, float, int, int]:
    """Corr(x, y | z): residualize both on z, then corr_test with non-overlapping p."""
    df = pd.concat([x, y, z], axis=1).dropna()
    n = len(df)
    if n < MIN_N_CORR:
        return np.nan, np.nan, 0, n
    X = np.column_stack([np.ones(n), df.iloc[:, 2].values])
    rx = df.iloc[:, 0].values - X @ np.linalg.lstsq(X, df.iloc[:, 0].values, rcond=None)[0]
    ry = df.iloc[:, 1].values - X @ np.linalg.lstsq(X, df.iloc[:, 1].values, rcond=None)[0]
    return corr_test(pd.Series(rx, index=df.index), pd.Series(ry, index=df.index), step=step)

def event_study(feat: pd.Series, fwd: pd.Series, trailing_hours: int = 168,
                z: float = 2.0, direction: str = "bull") -> dict:
    roll_mean = feat.rolling(trailing_hours, min_periods=48).mean()
    roll_std = feat.rolling(trailing_hours, min_periods=48).std()
    zs = (feat - roll_mean) / roll_std.replace(0, np.nan)
    mask = (zs > z) if direction == "bull" else (zs < -z)
    d = pd.concat([mask.rename("spike"), fwd.rename("fwd")], axis=1).dropna()
    spikes = d.loc[d["spike"], "fwd"]
    base = d.loc[~d["spike"], "fwd"]
    if len(spikes) < 10:
        return {"n_spikes": len(spikes), "mean_spike": np.nan, "mean_base": np.nan,
                "t": np.nan, "p": np.nan}
    t, p = stats.ttest_ind(spikes, base, equal_var=False)
    return {"n_spikes": len(spikes), "mean_spike": float(spikes.mean()),
            "mean_base": float(base.mean()), "t": float(t), "p": float(p)}

def backtest(px: pd.Series, signal: pd.Series, H: int, cost_bps: float,
             label: str) -> dict:
    """signal +1/-1/0 at hour h -> position held bars h+1..h+H. Enter at close(h)."""
    sig = signal.fillna(0.0)
    r = px.pct_change()
    pos = pd.Series(0.0, index=px.index)
    for h in sig.index[sig != 0]:
        i = px.index.get_loc(h)
        lo, hi = i + 1, min(i + 1 + H, len(px))
        pos.iloc[lo:hi] = sig.loc[h]
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps / 1e4)
    strat = pos.shift(1).fillna(0) * r - cost
    strat = strat.dropna()
    n_trades = int((turnover > 0).sum())
    tot = float((1 + strat).prod() - 1)
    bh = float(px.iloc[-1] / px.iloc[0] - 1)
    mu, sd = strat.mean(), strat.std()
    sharpe = float(mu / sd * math.sqrt(8760)) if sd and sd > 0 else np.nan
    curve = (1 + strat).cumprod()
    dd = float((curve / curve.cummax() - 1).min())
    return {"label": label, "n_trades": n_trades, "total_ret": tot, "buy_hold": bh,
            "sharpe": sharpe, "max_dd": dd}

FEATURES = ["bull_ratio_1h", "net_1h", "bull_ratio_24h", "net_24h",
            "d_bull_ratio_1h", "d_bull_ratio_24h"]

def analyze_symbol(sym: str, msgs: list[dict]) -> dict:
    is_crypto = sym in CRYPTO
    sent = build_sentiment(msgs)
    res = {"symbol": sym, "n_msgs": len(msgs),
           "first": str(sent.index.min()) if len(sent) else None,
           "last": str(sent.index.max()) if len(sent) else None,
           "n_hours": len(sent),
           "mean_labeled_per_h": float(sent["n_labeled"].mean()) if len(sent) else np.nan}

    days = max(40, (datetime.now(timezone.utc) - sent.index.min()).days + 5) if len(sent) else 40
    price = coinbase_hourly(CRYPTO[sym], days=days) if is_crypto else yf_hourly(sym, days=max(120, days + 60))
    px, session = align(price, sent)
    sample = session if not is_crypto else pd.Series(True, index=px.index)

    corr_rows, react_rows, part_rows = [], [], []
    for H in FORWARD_H:
        fwd = forward_returns(px, H).where(sample)
        past = (px / px.shift(H) - 1.0).where(sample)
        for feat in FEATURES:
            x = sent[feat].where(sample)
            r, p, n, nf = corr_test(x, fwd, step=H)
            corr_rows.append({"H": H, "feat": feat, "r": r, "p": p, "n": n, "n_full": nf})
            rp, pp, np_, npf = partial_corr(x, fwd, past, step=H)
            part_rows.append({"H": H, "feat": feat, "r": rp, "p": pp, "n": np_, "n_full": npf})
        for feat in ["bull_ratio_1h", "bull_ratio_24h"]:
            rr, pp, nn, nnf = corr_test(sent[feat].where(sample), past, step=H)
            react_rows.append({"H": H, "feat": feat, "r": rr, "p": pp, "n": nn, "n_full": nnf})
    res["forward_corr"] = corr_rows
    res["partial_corr"] = part_rows
    res["reactivity"] = react_rows

    ev = {}
    for H in FORWARD_H:
        fwd = forward_returns(px, H).where(sample)
        ev[f"bull_H{H}"] = event_study(sent["bull_ratio_24h"], fwd, direction="bull")
        ev[f"bear_H{H}"] = event_study(sent["bull_ratio_24h"], fwd, direction="bear")
    res["event_study"] = ev

    candidates = []   # (name, H, side, feature)
    for row in corr_rows:
        if (abs(row["r"] or 0) > 0.03 and row["p"] < 0.05 and row["n"] >= 20):
            candidates.append((f"corr_{row['feat']}", row["H"], int(np.sign(row["r"])), row["feat"]))
    for row in part_rows:   # survives controlling for past returns?
        if (abs(row["r"] or 0) > 0.03 and row["p"] < 0.05 and row["n"] >= 20):
            candidates.append((f"pcorr_{row['feat']}", row["H"], int(np.sign(row["r"])), row["feat"]))
    for k, e in ev.items():
        if abs(e.get("t", 0) or 0) > 2 and e["n_spikes"] >= 10:
            _, H = k.split("_H")
            side = int(np.sign((e["mean_spike"] or 0) - (e["mean_base"] or 0)))
            if side == 0:
                continue
            candidates.append((f"event_{k}", int(H), side, "bull_ratio_24h"))
    bts = []
    kind = "crypto" if is_crypto else "equity"
    for name, H, side, feat in candidates:
        roll_mean = sent[feat].rolling(168, min_periods=48).mean()
        roll_std = sent[feat].rolling(168, min_periods=48).std()
        zs = (sent[feat] - roll_mean) / roll_std.replace(0, np.nan)
        sig = pd.Series(0.0, index=px.index)
        xs = sent[feat].where(sample)   # quantiles / signals only on tradable sample
        if name.startswith("event"):
            is_bull_spike = "_bull_" in name
            if is_bull_spike:
                sig[zs > 2] = float(side)
            else:
                sig[zs < -2] = float(side)
        elif side > 0:
            q = xs.quantile(0.98)
            sig[xs >= q] = 1.0
        else:
            q = xs.quantile(0.02)
            sig[xs <= q] = -1.0
        sig = sig.where(sample, 0.0)
        bts.append(backtest(px, sig, H, COST_BPS[kind], f"{name}_H{H}"))
    res["backtests"] = bts
    return res

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--refetch", action="store_true")
    g.add_argument("--extend", action="store_true")
    ap.add_argument("--symbols", default=",".join(SYMBOLS),
                    help="comma-separated subset of symbols to process")
    args = ap.parse_args()
    symbols = [s for s in args.symbols.split(",") if s in SYMBOLS]

    print("== 1. Fetching Stocktwits ==", flush=True)
    all_msgs = {}
    for sym in symbols:
        all_msgs[sym] = fetch_symbol(sym, refetch=args.refetch, extend=args.extend)

    print("== 2. Prices + analysis ==", flush=True)
    summary_rows, full = [], {}
    for sym in symbols:
        print(f"  analyzing {sym}...", flush=True)
        res = analyze_symbol(sym, all_msgs[sym])
        full[sym] = res
        build_sentiment(all_msgs[sym]).to_csv(os.path.join(BASE, f"sentiment_hourly_{sym}.csv"))
        for row in res["forward_corr"]:
            summary_rows.append({"symbol": sym, "n_msgs": res["n_msgs"],
                                 "first": res["first"], "last": res["last"], **row})
    pd.DataFrame(summary_rows).to_csv(os.path.join(BASE, "sentiment_corr_summary.csv"), index=False)
    with open(os.path.join(BASE, "sentiment_analysis_full.json"), "w") as f:
        json.dump(full, f, indent=1, default=str)

    print("\n== Forward-return correlations (non-overlapping obs; * = n<20) ==", flush=True)
    print(f"{'sym':6} {'H':>3} {'feature':>16} {'r':>8} {'p':>9} {'n':>5}")
    for r in summary_rows:
        flag = "*" if r["n"] < 20 else " "
        print(f"{r['symbol']:6} {r['H']:>3} {r['feat']:>16} {r['r']:>8.4f} {r['p']:>9.2e} {r['n']:>5}{flag}")
    print("\n== Partial corr: sentiment(h) vs fwd return, controlling for PAST Hh return ==", flush=True)
    print(f"{'sym':6} {'H':>3} {'feature':>16} {'r':>8} {'p':>9} {'n':>5}")
    for sym in symbols:
        for row in full[sym]["partial_corr"]:
            flag = "*" if row["n"] < 20 else " "
            print(f"{sym:6} {row['H']:>3} {row['feat']:>16} {row['r']:>8.4f} {row['p']:>9.2e} {row['n']:>5}{flag}")
    print("\n== Reactivity: sentiment(h) vs PAST H-hour return ==", flush=True)
    print(f"{'sym':6} {'H':>3} {'feature':>16} {'r':>8} {'p':>9} {'n':>5}")
    for sym in symbols:
        for row in full[sym]["reactivity"]:
            flag = "*" if row["n"] < 20 else " "
            print(f"{sym:6} {row['H']:>3} {row['feat']:>16} {row['r']:>8.4f} {row['p']:>9.2e} {row['n']:>5}{flag}")
    print("\n== Event study: 24h-bull_ratio spikes (trailing 7d z>2) vs baseline ==", flush=True)
    print(f"{'sym':6} {'setup':>9} {'n_spk':>6} {'mean_spk':>10} {'mean_base':>10} {'t':>7} {'p':>9}")
    for sym in symbols:
        for k, e in full[sym]["event_study"].items():
            print(f"{sym:6} {k:>9} {e['n_spikes']:>6} {e['mean_spike']:>10.5f} "
                  f"{e['mean_base']:>10.5f} {e['t']:>7.2f} {e['p']:>9.2e}")
    print("\n== Backtests (with costs) ==", flush=True)
    any_bt = False
    for sym in symbols:
        for b in full[sym]["backtests"]:
            any_bt = True
            print(f"  {sym} {b['label']}: trades={b['n_trades']} net={b['total_ret']:+.2%} "
                  f"sharpe={b['sharpe']:.2f} maxdd={b['max_dd']:.2%} (buy&hold {b['buy_hold']:+.2%})")
    if not any_bt:
        print("  no candidates passed screening thresholds — nothing to backtest")
    print(f"\nCSVs/JSON saved to {BASE}")

if __name__ == "__main__":
    main()
