#!/usr/bin/env python3
"""SKEPTIC for Prometheus round 4: kill attempts against H11.

H11: Donchian/Turtle long-only breakout (N=20/55, M=N//2) on BTC daily,
reported +103.58%/+103.48% total, Sharpe 0.56/0.57, 10bps/side.

This script REPLICATES the original construction exactly (same data fetch,
same shifted rolling levels, same cost convention) and then tries to kill it:
 1. buy-and-hold on the exact same bars
 2. beta regression (R^2 of strategy on buy-hold)
 3. parameter grid N = 10,15,20,30,40,55,80,120
 4. entry timing: same-close vs next-open execution
 5. cost stress: 20bps / 40bps per side
 6. outlier dependence: drop best round-trip trade
 7. 200-day MA filter as the "just long in bull markets" control
 8. multiple-comparisons note (Bonferroni context, discussed in log)

Read-only research. No trading, no keys.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from bot.data.binance import fetch_klines  # noqa: E402

COST = 0.001  # original 10bps/side


def ann_sharpe(rets, ppy=365):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(ppy)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1)) if len(equity) else 0.0


def donchian_pos(cl, hi, lo, N):
    """Replicates round4_classics2.h11_donchian exactly: position decided at
    close t using trailing highs/lows strictly through t-1."""
    M = N // 2
    up = hi.rolling(N).max().shift(1)
    dn = lo.rolling(M).min().shift(1)
    pos = pd.Series(0, index=cl.index)
    p = 0
    for i in range(len(cl)):
        if p == 0 and cl.iloc[i] > up.iloc[i]:
            p = 1
        elif p == 1 and cl.iloc[i] < dn.iloc[i]:
            p = 0
        pos.iloc[i] = p
    return pos.astype(int)


def strat_returns_same_close(pos, ret_log, cost):
    """Original convention: pos decided at close t trades AT close t."""
    r = (pos.shift(1) * ret_log).fillna(0)
    r_net = r - (pos.diff().abs() > 0).astype(float) * cost
    return r_net


def strat_returns_next_open(pos, op, cl, cost):
    """Honest execution: signal decided at close t-1, executed at open t."""
    sig = pos.shift(1).fillna(0).astype(int)  # desired pos entering day t
    prev = pos.shift(2).fillna(0).astype(int)  # desired pos entering day t-1
    r = pd.Series(0.0, index=pos.index)
    log = np.log
    for i in range(1, len(pos)):
        s, pv = sig.iloc[i], prev.iloc[i]
        if s == 1 and pv == 1:
            r.iloc[i] = log(cl.iloc[i] / cl.iloc[i - 1])
        elif s == 1 and pv == 0:
            r.iloc[i] = log(cl.iloc[i] / op.iloc[i])  # enter at open
        elif s == 0 and pv == 1:
            r.iloc[i] = log(op.iloc[i] / cl.iloc[i - 1])  # exit at open
    r_net = r - (pos.diff().abs() > 0).astype(float) * cost
    return r_net


def round_trips(pos, cl, cost):
    """Extract round-trip trades: (entry_idx, exit_idx, net log return)."""
    d = pos.diff().fillna(0)
    entries = pos.index[d == 1].tolist()
    exits = pos.index[d == -1].tolist()
    trades = []
    ei = 0
    for ex in exits:
        # pair with the most recent entry before this exit
        while ei < len(entries) - 1 and entries[ei + 1] < ex:
            ei += 1
        if ei < len(entries) and entries[ei] < ex:
            en = entries[ei]
            ei += 1
            trades.append(
                (en, ex, float(np.log(cl.loc[ex] / cl.loc[en]) - 2 * cost))
            )
    # close any open position at the end
    if ei < len(entries):
        en = entries[ei]
        trades.append(
            (en, pos.index[-1], float(np.log(cl.iloc[-1] / cl.loc[en]) - cost))
        )
    return trades


def main():
    btc = fetch_klines("BTCUSDT", "1d", total=1500)
    btc.index = pd.to_datetime(btc.index, utc=True)
    cl, hi, lo, op = btc["close"], btc["high"], btc["low"], btc["open"]
    ret = np.log(cl / cl.shift(1))
    bh = ret.fillna(0)  # buy-and-hold log returns, same bars
    bh_tot = float(np.exp(bh.sum()) - 1)
    bh_sh = ann_sharpe(bh, 365)
    n = len(btc)
    print(f"bars={n} span={btc.index[0].date()}->{btc.index[-1].date()}")
    print(f"\n[1] BUY-HOLD same window: tot={bh_tot:+.2%} Sharpe={bh_sh:+.2f} "
          f"maxDD={max_dd(np.exp(bh.cumsum())):.2%}")

    for N in (20, 55):
        pos = donchian_pos(cl, hi, lo, N)
        r_net = strat_returns_same_close(pos, ret, COST)
        eq = np.exp(r_net.cumsum())
        ntr = int((pos.diff().abs() > 0).sum() / 2)
        tin = float(pos.mean())
        print(f"\nH11 N={N} replicate: trades={ntr} tot={eq.iloc[-1]-1:+.2%} "
              f"Sh={ann_sharpe(r_net,365):+.2f} maxDD={max_dd(eq):.2%} "
              f"time_in={tin:.1%}")
        print(f"    vs buy-hold: tot {eq.iloc[-1]-1:+.2%} vs {bh_tot:+.2%} "
              f"(ratio { (eq.iloc[-1])/(np.exp(bh.sum())) :.2f}), "
              f"Sh {ann_sharpe(r_net,365):+.2f} vs {bh_sh:+.2f}")

        # [2] beta representation
        df = pd.DataFrame({"s": r_net, "b": bh}).dropna()
        beta = float(np.cov(df["s"], df["b"])[0, 1] / np.var(df["b"]))
        resid = df["s"] - beta * df["b"]
        r2 = 1 - float(np.var(resid) / np.var(df["s"]))
        alpha_sh = ann_sharpe(resid, 365)
        print(f"[2] beta={beta:.2f} R^2={r2:.3f} residual(hedged) Sharpe={alpha_sh:+.2f}")

        # [4] entry timing
        r_open = strat_returns_next_open(pos, op, cl, COST)
        eqo = np.exp(r_open.cumsum())
        print(f"[4] next-open exec: tot={eqo.iloc[-1]-1:+.2%} "
              f"Sh={ann_sharpe(r_open,365):+.2f} "
              f"(delta tot {(eqo.iloc[-1]-eq.iloc[-1]):+.2%} vs same-close)")

        # [5] cost stress
        for c in (0.002, 0.004):
            rc = strat_returns_same_close(pos, ret, c)
            print(f"[5] cost {c*1e4:.0f}bps/side: tot={np.exp(rc.sum())-1:+.2%} "
                  f"Sh={ann_sharpe(rc,365):+.2f}")

        # [6] outliers
        trades = round_trips(pos, cl, COST)
        trets = sorted((t[2] for t in trades), reverse=True)
        best = trets[0] if trets else 0.0
        tot_log = float(r_net.sum())
        tot_no_best = float(np.exp(tot_log - best) - 1)
        print(f"[6] {len(trades)} round trips; best trade={np.exp(best)-1:+.2%} "
              f"(log {best:+.3f}); total ex-best={tot_no_best:+.2%} "
              f"(orig {eq.iloc[-1]-1:+.2%})")
        if len(trets) > 1:
            print(f"    top-3 trades (net): "
                  f"{', '.join(f'{np.exp(t)-1:+.1%}' for t in trets[:3])}")

        # [7] 200d MA control, same timing convention
        ma = cl.rolling(200).mean().shift(1)
        pos_ma = (cl > ma).astype(int).fillna(0)
        r_ma = strat_returns_same_close(pos_ma, ret, COST)
        eqm = np.exp(r_ma.cumsum())
        print(f"[7] 200d-MA long/flat control: tot={eqm.iloc[-1]-1:+.2%} "
              f"Sh={ann_sharpe(r_ma,365):+.2f} maxDD={max_dd(eqm):.2%} "
              f"time_in={float(pos_ma.mean()):.1%}")

    # [3] parameter grid
    print("\n[3] parameter grid (10bps/side, same-close):")
    print(f"{'N':>5} {'trades':>6} {'tot':>9} {'Sharpe':>7} {'maxDD':>8} "
          f"{'time_in':>8} {'H1':>6} {'H2':>6}")
    for N in (10, 15, 20, 30, 40, 55, 80, 120):
        pos = donchian_pos(cl, hi, lo, N)
        r_net = strat_returns_same_close(pos, ret, COST)
        eq = np.exp(r_net.cumsum())
        mid = len(r_net) // 2
        print(f"{N:>5} {int((pos.diff().abs()>0).sum()/2):>6} "
              f"{eq.iloc[-1]-1:>+8.2%} {ann_sharpe(r_net,365):>+7.2f} "
              f"{max_dd(eq):>8.2%} {float(pos.mean()):>8.1%} "
              f"{ann_sharpe(r_net.iloc[:mid],365):>+6.2f} "
              f"{ann_sharpe(r_net.iloc[mid:],365):>+6.2f}")

    # [8] context for multiple comparisons: how many params tried overall
    print("\n[8] multiple comparisons: H11 is ~hypothesis 11 in the loop; "
          "grid above tests 8 params x 2 halves. Bonferroni over ~30+ "
          "effective tests => significance threshold ~0.05/30. See log.")


if __name__ == "__main__":
    main()
