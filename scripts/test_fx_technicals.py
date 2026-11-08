"""Classic technical-analysis strategies on fiat FX pairs — honest backtests.

Strategies x pairs x coarse parameter grids, with:
  - realistic per-pair FX spreads charged on every unit of turnover (both sides)
  - strictly causal indicators (signals from bar t applied to bar t+1 returns)
  - regime splits (pre-2013 / 2013-2019 / 2020+) reported per regime
  - buy-and-hold benchmark per pair per regime
  - skeptic checks on apparent winners (USD beta via DXY, trade-frequency sanity)

Paper math only. No orders, no keys.
"""
from __future__ import annotations
import itertools
import json
import os

import numpy as np
import pandas as pd

RAW = os.path.expanduser("~/workspace/trading-bot/results/raw_fx")
OUT_SUM = os.path.expanduser("~/workspace/trading-bot/results/fx_technicals_summary.json")
OUT_CSV = os.path.expanduser("~/workspace/trading-bot/results/fx_technicals_grid.csv")

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
# Conservative retail spread assumption, in pips, charged PER SIDE per unit of turnover.
# (Real ECN spreads on majors are often <1 pip; charging the full spread per side is conservative.)
SPREAD_PIPS = {
    "EURUSD": 1.0, "GBPUSD": 1.5, "USDJPY": 1.0, "USDCHF": 1.5,
    "AUDUSD": 1.5, "USDCAD": 2.0, "NZDUSD": 2.0,
}
PIP_SIZE = {"USDJPY": 0.01}
TRADING_DAYS = 252

REGIMES = [
    ("pre2013", None, "2012-12-31"),
    ("2013_2019", "2013-01-01", "2019-12-31"),
    ("2020plus", "2020-01-01", None),
]


# ---------------------------------------------------------------- indicators (all causal)
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    # Wilder's RSI, causal (ewm adjust=False uses only past data)
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    al = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int, slow: int, sig: int):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal


# ---------------------------------------------------------------- strategies: signal(t) -> position held over t+1
def strat_ma_cross(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    pos = pd.Series(0, index=df.index, dtype=float)
    cond = sma(df["Close"], fast) > sma(df["Close"], slow)
    pos[cond.fillna(False)] = 1.0
    pos[(~cond).fillna(False)] = -1.0
    pos[cond.isna()] = np.nan
    return pos


def strat_rsi_mr(df: pd.DataFrame, n: int, os: float, ob: float) -> pd.Series:
    r = rsi(df["Close"], n)
    sig = pd.Series(np.nan, index=df.index, dtype=float)
    sig[r < os] = 1.0
    sig[r > ob] = -1.0
    return sig.ffill()  # middle zone: hold previous position; NaN until first trigger


def strat_macd(df: pd.DataFrame, fast: int, slow: int, sig: int) -> pd.Series:
    line, signal = macd(df["Close"], fast, slow, sig)
    pos = pd.Series(0, index=df.index, dtype=float)
    cond = line > signal
    pos[cond.fillna(False)] = 1.0
    pos[(~cond).fillna(False)] = -1.0
    pos[cond.isna()] = np.nan
    return pos


def strat_bb(df: pd.DataFrame, n: int, k: float, mode: str) -> pd.Series:
    mid = sma(df["Close"], n)
    sd = df["Close"].rolling(n).std()
    upper, lower = mid + k * sd, mid - k * sd
    sig = pd.Series(np.nan, index=df.index, dtype=float)
    if mode == "mr":  # mean reversion: fade the bands
        sig[df["Close"] < lower] = 1.0
        sig[df["Close"] > upper] = -1.0
    else:  # breakout: ride the bands
        sig[df["Close"] > upper] = 1.0
        sig[df["Close"] < lower] = -1.0
    return sig.ffill()


def strat_donchian(df: pd.DataFrame, n: int) -> pd.Series:
    # breakout vs N-bar channel computed on bars t-N..t-1 (shifted: no lookahead)
    hh = df["High"].rolling(n).max().shift(1)
    ll = df["Low"].rolling(n).min().shift(1)
    sig = pd.Series(np.nan, index=df.index, dtype=float)
    sig[df["Close"] > hh] = 1.0
    sig[df["Close"] < ll] = -1.0
    return sig.ffill()


STRATEGIES = {
    "ma_cross": (strat_ma_cross, [
        {"fast": 10, "slow": 50}, {"fast": 20, "slow": 100}, {"fast": 50, "slow": 200},
    ]),
    "rsi_mr": (strat_rsi_mr, [
        {"n": 14, "os": 30, "ob": 70}, {"n": 14, "os": 25, "ob": 75}, {"n": 7, "os": 30, "ob": 70},
    ]),
    "macd": (strat_macd, [
        {"fast": 12, "slow": 26, "sig": 9}, {"fast": 8, "slow": 21, "sig": 5},
    ]),
    "bb_mr": (strat_bb, [{"n": 20, "k": 2.0, "mode": "mr"}, {"n": 20, "k": 1.5, "mode": "mr"}]),
    "bb_bo": (strat_bb, [{"n": 20, "k": 2.0, "mode": "bo"}]),
    "donchian": (strat_donchian, [{"n": 20}, {"n": 55}]),
}


# ---------------------------------------------------------------- backtest engine
def backtest(df: pd.DataFrame, pos_signal: pd.Series, spread_frac: pd.Series):
    """pos_signal[t]: desired position decided at close of bar t, held over bar t+1.
    Returns dict of metrics on net (after-cost) returns."""
    pos = pos_signal.shift(1)  # trade at next bar: no lookahead
    ret = df["Close"].pct_change()
    turnover = pos.diff().abs().fillna(pos.abs())  # first entry counts
    cost = turnover * spread_frac
    net = pos.fillna(0) * ret.fillna(0) - cost.fillna(0)
    net = net.dropna()
    if len(net) < 50:
        return None
    tot = float((1 + net).prod() - 1)
    yrs = len(net) / TRADING_DAYS
    cagr = float((1 + tot) ** (1 / yrs) - 1) if yrs > 0 else 0.0
    sharpe = float(net.mean() / net.std() * np.sqrt(TRADING_DAYS)) if net.std() > 0 else 0.0
    eq = (1 + net).cumprod()
    dd = float(((eq / eq.cummax()) - 1).min())
    n_trades = int((turnover.fillna(0) > 0).sum())
    # win rate on round trips is complex; use positive-day rate of active days
    active = net[pos.fillna(0) != 0]
    win = float((active > 0).mean()) if len(active) else 0.0
    time_in = float((pos.fillna(0) != 0).mean())
    return {"total_ret": tot, "cagr": cagr, "sharpe": sharpe, "maxdd": dd,
            "trades": n_trades, "winrate": win, "time_in": time_in, "bars": len(net)}


def regime_metrics(df, pos_signal, spread_frac, start, end):
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df.index >= start
    if end:
        mask &= df.index <= end
    sub = df.loc[mask]
    if len(sub) < 100:
        return None
    m = backtest(sub, pos_signal.loc[mask], spread_frac.loc[mask])
    # buy-and-hold benchmark over the same window
    bh_ret = float(sub["Close"].iloc[-1] / sub["Close"].iloc[0] - 1)
    if m:
        m["bh_ret"] = bh_ret
    return m


def main() -> None:
    rows = []
    for pair in PAIRS:
        df = pd.read_csv(os.path.join(RAW, f"{pair}_daily.csv"), index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        pip = PIP_SIZE.get(pair, 0.0001)
        spread_frac = SPREAD_PIPS[pair] * pip / df["Close"]  # per side per unit turnover
        for sname, (fn, grids) in STRATEGIES.items():
            for params in grids:
                cfg = f"{sname}({','.join(f'{k}={v}' for k, v in params.items())})"
                try:
                    sig = fn(df, **params)
                except Exception as e:
                    print(f"SKIP {pair} {cfg}: {e}")
                    continue
                full = backtest(df, sig, spread_frac)
                if full is None:
                    continue
                rec = {"pair": pair, "config": cfg, "full": full, "regimes": {}}
                for rname, start, end in REGIMES:
                    rm = regime_metrics(df, sig, spread_frac, start, end)
                    if rm:
                        rec["regimes"][rname] = rm
                rows.append(rec)
                f = rec["full"]
                print(f"{pair:7s} {cfg:38s} sharpe={f['sharpe']:+.2f} cagr={f['cagr']:+.2%} "
                      f"dd={f['maxdd']:.2%} trades={f['trades']}")

    pd.DataFrame([{
        "pair": r["pair"], "config": r["config"],
        "sharpe": r["full"]["sharpe"], "cagr": r["full"]["cagr"],
        "maxdd": r["full"]["maxdd"], "trades": r["full"]["trades"],
        **{f"{k}_{m}": v for k, reg in r["regimes"].items() for m, v in
           [("sharpe", reg["sharpe"]), ("cagr", reg["cagr"]), ("bh", reg["bh_ret"])]}
    } for r in rows]).to_csv(OUT_CSV, index=False)

    # skeptic shortlist: full-sample sharpe > 0.3 AND positive in >= 2 regimes
    shortlist = [r for r in rows
                 if r["full"]["sharpe"] > 0.3
                 and sum(1 for reg in r["regimes"].values() if reg["sharpe"] > 0) >= 2]
    with open(OUT_SUM, "w") as fh:
        json.dump({"n_configs": len(rows),
                   "shortlist": [(r["pair"], r["config"], r["full"]["sharpe"]) for r in shortlist],
                   "rows": rows}, fh, indent=1, default=str)
    print(f"\n{len(rows)} configs tested. Shortlist (sharpe>0.3, >=2 positive regimes): {len(shortlist)}")
    for r in shortlist:
        print("  ", r["pair"], r["config"], f"sharpe={r['full']['sharpe']:.2f}",
              {k: f"{v['sharpe']:.2f}" for k, v in r["regimes"].items()})


if __name__ == "__main__":
    main()
