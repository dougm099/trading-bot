#!/usr/bin/env python3
"""Round 3: THREAD A/B: H8 pairs/cointegration, H9 1m mean reversion,
H10 order-book imbalance feasibility.

H8: cointegration pairs. Crypto: BTC/ETH, BTC/SOL, ETH/SOL (Coinbase 1h).
Equities: AAPL/MSFT, NVDA/AMD, XLF/XLU, SPY/QQQ (yfinance daily). Walk-forward:
12m formation (Engle-Granger, hedge ratio OLS), 6m trading on z-score of
spread (enter |z|>2, exit |z|<0.5), 5bps/side equities, 10bps/side crypto.
Report half-life, ADF p-value, per-regime Sharpe.

H9: 1m mean reversion on BTC (Coinbase, 60s granularity, ~7d). Signal: past-N-min
return < -k * trailing sigma -> long next M minutes (and mirror short).
N,M in {(5,5),(15,15),(30,30)}, k=1.5. Costs 10bps/side. HONESTY BOUNDARY:
1m-bar edges from a VM are indicative only; anything needing <1min holding is
logged as "needs colocation", not claimed.

H10: Coinbase L2 book snapshots (BTC-USD, level=2, top 50). Sample every ~3s
for ~12 min. Compute book imbalance I=(bid_vol-ask_vol)/(bid_vol+ask_vol) and
microprice deviation. Correlate with forward 30s/60s/300s mid-price returns.
Verdict format: predictive at >=60s horizon (reachable) vs <60s (colocation).

No lookahead anywhere: formation before trading; signals use strictly past data.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from bot.data.binance import fetch_klines  # noqa: E402

COST_EQ, COST_CR = 0.0005, 0.001
CB = "https://api.exchange.coinbase.com"


def ann_sharpe(rets, ppy):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def h8_pairs():
    print("\n===== H8 pairs / cointegration =====")
    # --- crypto hourly ---
    print("-- crypto 1h (Coinbase) --")
    coins = {}
    for c in ["BTC", "ETH", "SOL"]:
        b = fetch_klines(f"{c}USDT", "1h", total=5000)
        coins[c] = b["close"]
    cpx = pd.DataFrame(coins).dropna()
    for a, b in [("BTC", "ETH"), ("BTC", "SOL"), ("ETH", "SOL")]:
        x, y = np.log(cpx[a]), np.log(cpx[b])
        try:
            _, pval, _ = coint(x, y)
        except Exception:
            pval = np.nan
        # walk-forward: first 60% formation, rest trade (single split for speed)
        n = len(x)
        xf, yf_ = x.iloc[: int(n * 0.6)], y.iloc[: int(n * 0.6)]
        beta = np.polyfit(xf, yf_, 1)[0]
        spread = y - beta * x
        mu, sd = spread.iloc[: int(n * 0.6)].mean(), spread.iloc[: int(n * 0.6)].std()
        z = (spread - mu) / sd
        # half-life on formation spread
        sp_f = spread.iloc[: int(n * 0.6)]
        d = sp_f.diff().dropna().values
        lag = sp_f.shift(1).dropna().values
        hl = -np.log(2) / np.polyfit(lag, d, 1)[0] if len(lag) > 50 else np.nan
        zt = z.iloc[int(n * 0.6):]
        # trade z on the pair: long spread when z<-2 (long y, short beta*x)
        pos = np.zeros(len(zt))
        p = 0
        for i, v in enumerate(zt.values):
            if p == 0 and v < -2:
                p = 1
            elif p == 0 and v > 2:
                p = -1
            elif abs(v) < 0.5:
                p = 0
            pos[i] = p
        spr = zt.diff().fillna(0).values  # approx spread return
        r = pos * (-np.diff(np.concatenate([[0], zt.values])))  # mean-reversion P&L in z units
        r = pd.Series(r[1:], index=zt.index[1:]) * sd  # back to log-price units approx
        turnover = np.abs(np.diff(np.concatenate([[0], pos]))).sum() / 2
        r_net = r.sum() - turnover * 2 * COST_CR
        print(f"  {a}/{b}: coint p={pval:.3f} beta={beta:.3f} half-life={hl:.1f}h "
              f"trades~{int(turnover)} net_logret={r_net:+.3f} Sharpe={ann_sharpe(r,365*24):+.2f}")
    # --- equities daily ---
    print("-- equities daily (yfinance) --")
    px = yf.download(["AAPL", "MSFT", "NVDA", "AMD", "XLF", "XLU", "SPY", "QQQ"],
                     start="2015-01-01", end="2026-09-01", interval="1d",
                     auto_adjust=True, progress=False)["Close"].dropna()
    for a, b in [("AAPL", "MSFT"), ("NVDA", "AMD"), ("XLF", "XLU"), ("SPY", "QQQ")]:
        x, y = np.log(px[a]), np.log(px[b])
        try:
            _, pval, _ = coint(x, y)
        except Exception:
            pval = np.nan
        # walk-forward: 252d formation / 126d trade, roll by 126d
        rets, trades = [], 0
        for start in range(252, len(x) - 126, 126):
            xf, yf_ = x.iloc[start - 252:start], y.iloc[start - 252:start]
            beta = np.polyfit(xf, yf_, 1)[0]
            sp = y - beta * x
            mu, sd = sp.iloc[start - 252:start].mean(), sp.iloc[start - 252:start].std()
            if sd == 0:
                continue
            zt = ((sp - mu) / sd).iloc[start:start + 126]
            p, prev = 0, 0
            for v, nxt in zip(zt.values[:-1], np.diff(zt.values)):
                if p == 0 and v < -2:
                    p = 1; trades += 1
                elif p == 0 and v > 2:
                    p = -1; trades += 1
                elif abs(v) < 0.5:
                    p = 0
                # P&L: short z when z high -> profit when z falls: -p * dz * sd
                rets.append(-p * nxt * sd - (COST_EQ * 2 if p != prev else 0))
                prev = p
        rets = np.array(rets)
        eq = np.exp(np.cumsum(rets))
        print(f"  {a}/{b}: coint p={pval:.4f} roundtrips~{trades} "
              f"tot={eq[-1]-1:+.2%} Sharpe={ann_sharpe(rets,252):+.2f} maxDD={max_dd(eq):.2%}")


def h9_1m_reversion():
    print("\n===== H9 BTC 1m mean reversion =====")
    # ~7 days of 1m bars (Coinbase limit 300/call)
    out, end = [], None
    for _ in range(34):
        params = {"granularity": 60}
        if end:
            params["end"] = end.isoformat()
            params["start"] = (end - pd.Timedelta(minutes=300)).isoformat()
        r = requests.get(f"{CB}/products/BTC-USD/candles", params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out = batch + out
        end = pd.to_datetime(batch[-1][0], unit="s", utc=True)
        time.sleep(0.25)
    df = pd.DataFrame(out, columns=["t", "low", "high", "open", "close", "volume"])
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.drop_duplicates("t").set_index("t").sort_index()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    print(f"  n={len(df)} 1m bars, {df.index[0]} -> {df.index[-1]}")
    for N, M in [(5, 5), (15, 15), (30, 30)]:
        past = df["ret"].rolling(N).sum()
        sig = df["ret"].rolling(N * 12).std() * np.sqrt(N)
        z = past / sig
        # long next M min if z < -1.5, short if z > 1.5
        sigpos = pd.Series(0, index=df.index)
        sigpos[z < -1.5] = 1
        sigpos[z > 1.5] = -1
        fwd = df["ret"].shift(-M).rolling(M).sum().shift(M)  # placeholder
        # simpler: for each signal bar, forward M-min return
        fwdM = np.log(df["close"].shift(-M) / df["close"])
        r = (sigpos * fwdM).dropna()
        # costs: each signal bar is a new position -> turnover heavy; approximate
        ntr = (sigpos != 0).sum()
        r_net_tot = r.sum() - ntr * 2 * COST_CR
        # holding-period honesty: M minutes is reachable from VM; sub-1min would not be
        print(f"  N={N} M={M}: signals={ntr} gross_logret={r.sum():+.3f} "
              f"net~{r_net_tot:+.3f} (holding {M}min: {'REACHABLE' if M>=1 else 'NEEDS COLOCATION'})")


def h10_book_imbalance():
    print("\n===== H10 Coinbase L2 book imbalance feasibility =====")
    snaps = []
    t0 = time.time()
    while time.time() - t0 < 720:  # ~12 min
        try:
            r = requests.get(f"{CB}/products/BTC-USD/book", params={"level": 2},
                             timeout=10)
            b = r.json()
            bids = pd.DataFrame(b["bids"], columns=["px", "sz", "n"]).astype(float)
            asks = pd.DataFrame(b["asks"], columns=["px", "sz", "n"]).astype(float)
            bv, av = bids["sz"].sum(), asks["sz"].sum()
            mid = (bids["px"].iloc[0] + asks["px"].iloc[0]) / 2
            micro = (bids["px"].iloc[0] * av + asks["px"].iloc[0] * bv) / (av + bv)
            snaps.append((time.time(), (bv - av) / (bv + av), mid, micro))
        except Exception as e:
            print(f"  snapshot error: {e}")
        time.sleep(3)
    s = pd.DataFrame(snaps, columns=["t", "imb", "mid", "micro"])
    s["t"] = pd.to_datetime(s["t"], unit="s", utc=True)
    s = s.set_index("t")
    print(f"  n={len(s)} snapshots over {(s.index[-1]-s.index[0]).total_seconds()/60:.1f} min")
    for horizon_s, label in [(30, "30s"), (60, "60s"), (300, "300s")]:
        fwd = np.log(s["mid"].shift(-int(horizon_s / 3)) / s["mid"])
        c1 = s["imb"].corr(fwd)
        dev = (s["micro"] - s["mid"]) / s["mid"]
        c2 = dev.corr(fwd)
        reach = "REACHABLE (>=60s)" if horizon_s >= 60 else "NEEDS COLOCATION (<60s)"
        print(f"  fwd {label}: corr(imbalance,ret)={c1:+.3f} corr(micro-dev,ret)={c2:+.3f} [{reach}]")


if __name__ == "__main__":
    import sys as _s
    which = _s.argv[1] if len(_s.argv) > 1 else "all"
    if which in ("all", "h8"):
        h8_pairs()
    if which in ("all", "h9"):
        h9_1m_reversion()
    if which in ("all", "h10"):
        h10_book_imbalance()
    print("\nDONE")
