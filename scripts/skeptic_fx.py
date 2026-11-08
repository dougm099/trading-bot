"""Skeptic pass on the two FX technicals shortlist candidates.
Checks: return concentration, USD-beta (DXY correlation), vs buy-and-hold excess,
trade-frequency coin-flip baseline, and where in time the P&L comes from.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.expanduser("~/workspace/trading-bot/scripts"))
from test_fx_technicals import strat_rsi_mr, strat_bb, SPREAD_PIPS, PIP_SIZE, REGIMES  # noqa

RAW = os.path.expanduser("~/workspace/trading-bot/results/raw_fx")

CANDIDATES = [
    ("GBPUSD", strat_rsi_mr, {"n": 7, "os": 30, "ob": 70}),
    ("USDCHF", strat_bb, {"n": 20, "k": 2.0, "mode": "mr"}),
]


def load(pair):
    df = pd.read_csv(os.path.join(RAW, f"{pair}_daily.csv"), index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df


def net_returns(df, sig, pair):
    pip = PIP_SIZE.get(pair, 0.0001)
    spread_frac = SPREAD_PIPS[pair] * pip / df["Close"]
    pos = sig.shift(1)
    ret = df["Close"].pct_change()
    turnover = pos.diff().abs().fillna(pos.abs())
    return (pos.fillna(0) * ret.fillna(0) - (turnover * spread_frac).fillna(0)).dropna()


def main():
    # DXY for USD-beta check (DX-Y.NYB, daily, as far back as available)
    dxy = yf.Ticker("DX-Y.NYB").history(period="max", auto_adjust=True)[["Close"]].dropna()
    dxy.index = pd.to_datetime(dxy.index, utc=True).tz_localize(None)
    dxy_ret = dxy["Close"].pct_change().rename("DXY")

    for pair, fn, params in CANDIDATES:
        df = load(pair)
        sig = fn(df, **params)
        net = net_returns(df, sig, pair)
        bh = df["Close"].pct_change().reindex(net.index).fillna(0)
        print(f"\n{'='*70}\n{pair} {fn.__name__}{params}\n{'='*70}")

        # 1. return concentration: top-5-day share of total P&L
        tot = float((1 + net).prod() - 1)
        top5 = net.nlargest(5).sum()
        # approximate additive contribution
        print(f"total net return: {tot:+.2%} over {len(net)} days")
        print(f"sum of 5 best days: {top5:+.4f} (additive)")

        # 2. per-regime excess vs buy-and-hold
        for rname, start, end in REGIMES:
            m = (net.index >= (start or "1900-01-01")) & (net.index <= (end or "2100-01-01"))
            sub, bsub = net[m], bh[m]
            s_ex = float((1 + sub).prod() - 1) - float((1 + bsub).prod() - 1)
            sh = float(sub.mean() / sub.std() * np.sqrt(252)) if sub.std() > 0 else 0
            print(f"  regime {rname:10s}: strat={(1+sub).prod()-1:+.2%} bh={(1+bsub).prod()-1:+.2%} "
                  f"excess={s_ex:+.2%} sharpe={sh:+.2f} days={m.sum()}")

        # 3. USD beta: correlation with DXY
        j = pd.concat([net, dxy_ret], axis=1, join="inner").dropna()
        print(f"DXY return correlation: {j.iloc[:,0].corr(j['DXY']):+.3f} "
              f"(n={len(j)}); USD-direction check")

        # 4. coin-flip baseline: same trade frequency, random direction, after costs
        rng = np.random.default_rng(42)
        pos = sig.shift(1).fillna(0)
        n_sims, wins = 200, 0
        simrets = []
        for _ in range(n_sims):
            rpos = pd.Series(rng.choice([-1.0, 1.0], size=len(pos)), index=pos.index)
            # match activity: only trade when the real strategy is active
            rpos = rpos * (pos != 0).astype(float)
            turn = rpos.diff().abs().fillna(rpos.abs())
            pip = PIP_SIZE.get(pair, 0.0001)
            sf = SPREAD_PIPS[pair] * pip / df["Close"]
            rnet = (rpos * df["Close"].pct_change().fillna(0) - (turn * sf).fillna(0)).dropna()
            sr = float(rnet.mean() / rnet.std() * np.sqrt(252)) if rnet.std() > 0 else 0
            simrets.append(sr)
            if sr >= float(net.mean() / net.std() * np.sqrt(252)):
                wins += 1
        real_sh = float(net.mean() / net.std() * np.sqrt(252))
        print(f"real Sharpe {real_sh:+.2f} vs {n_sims} random-direction sims "
              f"(same activity mask): mean sim Sharpe {np.mean(simrets):+.2f}, "
              f"P(sim >= real) = {wins/n_sims:.2f}")

        # 5. where does the money come from? best/worst 30-day windows
        roll = (1 + net).rolling(30).apply(lambda x: x.prod() - 1, raw=True)
        print(f"best 30d window: {roll.max():+.2%} ending {roll.idxmax().date()}; "
              f"worst 30d: {roll.min():+.2%} ending {roll.idxmin().date()}")
        # yearly breakdown
        y = net.groupby(net.index.year).apply(lambda x: float((1 + x).prod() - 1))
        print("yearly net returns:")
        print("  " + " ".join(f"{yr}:{r:+.0%}" for yr, r in y.items()))


if __name__ == "__main__":
    main()
