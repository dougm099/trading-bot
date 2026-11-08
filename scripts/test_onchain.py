"""Test on-chain/crypto-fundamental signals with kill-or-keep discipline.

Signals (all daily, no lookahead — signals lagged appropriately):
  S1 stablecoin mcap growth -> BTC forward returns ("dry powder")
  S2 SSR oscillator (BTC mcap / stablecoin mcap), z-scored
  S3 exchange netflows (Coin Metrics, lagged 2d for flash revisions) -> forward returns
  S4 cross-venue funding gap (Hyperliquid vs OKX) -> delta-neutral basis carry

Costs: 10 bps/side crypto (S4: 2 legs each side). Split-sample regimes.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

RAW = os.path.expanduser("~/workspace/trading-bot/results/raw_onchain")
COST = 0.001  # 10 bps / side
ANN = 365

def load():
    sc = pd.read_csv(f"{RAW}/stablecoin_mcap_daily.csv", parse_dates=["date"])
    # DeFiLlama aggregate stablecoin history is incomplete before 2020
    # (2017 value $110k vs actual ~$1B) -> restrict to the clean era.
    sc = sc[sc["date"] >= "2020-01-01"].reset_index(drop=True)
    cm = pd.read_csv(f"{RAW}/coinmetrics_btc_daily.csv", parse_dates=["date"])
    btc = pd.read_csv(f"{RAW}/btc_daily.csv", parse_dates=["date"])
    bmx = pd.read_csv(f"{RAW}/bitmex_funding.csv")
    bmx["ts"] = pd.to_datetime(bmx["ts"], utc=True)
    hl = pd.read_csv(f"{RAW}/hl_funding.csv")
    hl["ts"] = pd.to_datetime(hl["ts"], utc=True, format="mixed")
    return sc, cm, btc, bmx, hl

def regime_split(dates):
    """Chronological regime blocks."""
    d = pd.to_datetime(dates)
    def tag(x):
        y = x.year
        if y <= 2019: return "2018-19 (bear/recovery)"
        if y <= 2021: return "2020-21 (bull)"
        if y == 2022: return "2022 (bear)"
        return "2023-26 (recovery/bull)"
    return d.map(tag)

def backtest_stats(ret: pd.Series, cost_trades: pd.Series | None = None):
    r = ret.fillna(0.0)
    if cost_trades is not None:
        r = r - cost_trades.fillna(0.0)
    tot = float((1 + r).prod() - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"return": tot, "sharpe": sharpe, "maxDD": dd, "n": len(r)}

def regime_table(dates, strat_ret, bh_ret, cost_series, name):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "strat": strat_ret,
                       "bh": bh_ret, "cost": cost_series}).dropna()
    df["regime"] = regime_split(df["date"])
    rows = []
    for reg, g in df.groupby("regime", sort=True):
        s = backtest_stats(g["strat"], g["cost"])
        b = backtest_stats(g["bh"], None)
        rows.append({"signal": name, "regime": reg,
                     "strat_ret": round(s["return"], 4), "strat_sharpe": round(s["sharpe"], 2),
                     "bh_ret": round(b["return"], 4), "bh_sharpe": round(b["sharpe"], 2),
                     "n_days": len(g)})
    return pd.DataFrame(rows)

def quantile_backtest(signal: pd.Series, fwd_ret: pd.Series, dates,
                      q=0.25, long_top=True, cost=COST):
    """Long when signal in top/bottom quantile (expanding quantile, lagged signal)."""
    sig = signal.shift(1)  # no lookahead: decide at close t, trade t+1
    pos = pd.Series(0.0, index=sig.index)
    thresh = sig.expanding(min_periods=252).quantile(1 - q if long_top else q)
    pos[sig >= thresh if long_top else sig <= thresh] = 1.0
    pos = pos.shift(1).fillna(0.0)  # position held during day t+1
    trades = pos.diff().abs()
    strat_ret = pos * fwd_ret
    return strat_ret, trades * cost, pos

# ---------------------------------------------------------------- S1
def test_s1(sc, btc):
    df = pd.merge(sc, btc[["date", "close"]], on="date", how="inner").sort_values("date")
    df = df.dropna(subset=["stable_mcap_usd", "close"]).reset_index(drop=True)
    df["sc_g30"] = df["stable_mcap_usd"].pct_change(30)
    df["fwd30"] = df["close"].pct_change(30).shift(-30)
    df["fwd7"] = df["close"].pct_change(7).shift(-7)
    sub = df.dropna(subset=["sc_g30", "fwd30"])
    out = {"n": len(sub),
           "corr_scg30_fwd30": float(sub["sc_g30"].corr(sub["fwd30"])),
           "corr_scg30_fwd7": float(sub["sc_g30"].corr(sub["fwd7"]))}
    # backtest: long BTC when 30d stablecoin growth in top quartile (expanding)
    strat, cost_s, pos = quantile_backtest(df.set_index("date")["sc_g30"],
                                           df.set_index("date")["close"].pct_change(),
                                           df["date"], q=0.25, long_top=True)
    idx = df.set_index("date").index
    bh = df.set_index("date")["close"].pct_change()
    reg = regime_table(idx, strat.reindex(idx).fillna(0), bh.reindex(idx).fillna(0),
                       cost_s.reindex(idx).fillna(0), "S1 scap-growth long-BTC")
    out["time_in"] = float((pos > 0).mean())
    out["trades"] = int((pos.diff().abs() > 0).sum())
    return out, reg

# ---------------------------------------------------------------- S2
def test_s2(sc, cm, btc):
    df = pd.merge(sc, cm[["date", "CapMrktCurUSD"]], on="date", how="inner")
    df = pd.merge(df, btc[["date", "close"]], on="date", how="inner").sort_values("date")
    df = df.dropna(subset=["stable_mcap_usd", "CapMrktCurUSD"]).reset_index(drop=True)
    df["ssr"] = df["CapMrktCurUSD"] / df["stable_mcap_usd"]
    df["log_ssr"] = np.log(df["ssr"])
    roll = df["log_ssr"].expanding(min_periods=365)
    df["ssr_z"] = (df["log_ssr"] - roll.mean()) / roll.std()
    sig = df.set_index("date")["ssr_z"].shift(1)
    # long when SSR cheap (z < -1), flat when expensive (z > +1), else hold prior
    pos = pd.Series(0.0, index=sig.index)
    pos[sig < -1.0] = 1.0
    pos = pos.replace(0.0, np.nan)
    # exit long when z > +1
    long_mask = (sig < -1.0)
    exit_mask = (sig > 1.0)
    p = pd.Series(0.0, index=sig.index)
    cur = 0.0
    for i in sig.index:
        if long_mask.loc[i]: cur = 1.0
        elif exit_mask.loc[i]: cur = 0.0
        p.loc[i] = cur
    pos = p.shift(1).fillna(0.0)
    ret = df.set_index("date")["close"].pct_change()
    strat = pos * ret
    cost_s = pos.diff().abs() * COST
    idx = df.set_index("date").index
    reg = regime_table(idx, strat.reindex(idx).fillna(0), ret.reindex(idx).fillna(0),
                       cost_s.reindex(idx).fillna(0), "S2 SSR-oscillator")
    sub = df.dropna(subset=["ssr_z"]).copy()
    sub["fwd30"] = sub["close"].pct_change(30).shift(-30)
    out = {"n": len(sub), "corr_ssrz_fwd30": float(sub["ssr_z"].corr(sub["fwd30"])),
           "time_in": float((pos > 0).mean()), "trades": int((pos.diff().abs() > 0).sum()),
           "ssr_min": float(df["ssr"].min()), "ssr_max": float(df["ssr"].max())}
    return out, reg

# ---------------------------------------------------------------- S3
def test_s3(cm, btc):
    df = pd.merge(cm, btc[["date", "close"]], on="date", how="inner").sort_values("date")
    df = df.dropna(subset=["FlowInExNtv", "FlowOutExNtv"]).reset_index(drop=True)
    df["netflow"] = df["FlowInExNtv"] - df["FlowOutExNtv"]
    roll = df["netflow"].expanding(min_periods=180)
    df["nf_z"] = (df["netflow"] - roll.mean()) / roll.std()
    # lag 2 extra days: Coin Metrics marks recent rows 'flash' (revised ~36-40h later)
    sig = df.set_index("date")["nf_z"].shift(2)
    df["fwd7"] = df["close"].pct_change(7).shift(-7)
    sub = df.dropna(subset=["nf_z", "fwd7"])
    out = {"n": len(sub), "corr_nfz_fwd7": float(sub["nf_z"].corr(sub["fwd7"])),
           "flash_lag_days": 2}
    # backtest: inflows = selling pressure -> short when z > 1.5, long when z < -1.5
    pos = pd.Series(0.0, index=sig.index)
    pos[sig > 1.5] = -1.0
    pos[sig < -1.5] = 1.0
    pos = pos.shift(1).fillna(0.0)
    ret = df.set_index("date")["close"].pct_change()
    strat = pos * ret
    cost_s = pos.diff().abs() * COST
    idx = df.set_index("date").index
    reg = regime_table(idx, strat.reindex(idx).fillna(0), ret.reindex(idx).fillna(0),
                       cost_s.reindex(idx).fillna(0), "S3 exchange-netflow L/S")
    out["time_in"] = float((pos != 0).mean())
    out["trades"] = int((pos.diff().abs() > 0).sum())
    return out, reg

# ---------------------------------------------------------------- S4
def annualize_funding(df, interval_hours):
    d = df.copy().sort_values("ts")
    d["ann"] = d["funding_rate"] * (ANN * 24 / interval_hours)
    return d

def test_s4(bmx, hl, btc):
    # detect intervals
    hl_iv = float(hl.sort_values("ts")["ts"].diff().median() / pd.Timedelta(hours=1))
    bmx_iv = float(bmx.sort_values("ts")["ts"].diff().median() / pd.Timedelta(hours=1))
    hl_a = annualize_funding(hl, hl_iv).set_index("ts")["ann"]
    bmx_a = annualize_funding(bmx, bmx_iv).set_index("ts")["ann"]
    # align to common 8h grid
    grid = pd.date_range(max(hl_a.index.min(), bmx_a.index.min()).floor("8h"),
                         min(hl_a.index.max(), bmx_a.index.max()).ceil("8h"),
                         freq="8h", tz="UTC")
    h = hl_a.reindex(grid, method="ffill", limit=3)
    o = bmx_a.reindex(grid, method="ffill", limit=3)
    gap = (h - o).dropna()  # >0: HL pays more to longs => short HL / long BitMEX
    TH = 0.05  # 5% annualized gap threshold
    pos = pd.Series(0.0, index=gap.index)  # +1 = short HL / long BitMEX (collect gap)
    pos[gap > TH] = 1.0
    pos[gap < -TH] = -1.0
    pos = pos.shift(1).fillna(0.0)  # trade next interval on published funding
    # P&L per interval = pos * gap * (interval fraction of year), minus 2 legs * 10bps on flips
    dt_years = 8 / (ANN * 24)
    pnl = pos * gap.shift(1).fillna(0) * dt_years
    flips = pos.diff().abs()
    cost_s = flips * 2 * COST  # each flip changes 2 legs
    strat_ret = pnl - cost_s
    # benchmark: T-bill ~4.5%/yr over same window
    tbill = pd.Series(0.045 * dt_years, index=gap.index)
    stats = backtest_stats(strat_ret, None)
    tb_stats = backtest_stats(tbill, None)
    out = {"n_intervals": len(gap), "hl_interval_h": hl_iv, "bmx_interval_h": bmx_iv,
           "threshold_ann": TH, "time_in": float((pos != 0).mean()),
           "flips": int((flips > 0).sum()),
           "strat_total": round(stats["return"], 4), "strat_sharpe": round(stats["sharpe"], 2),
           "strat_maxDD": round(stats["maxDD"], 4),
           "tbill_total": round(tb_stats["return"], 4),
           "mean_abs_gap_ann": round(float(gap.abs().mean()), 4),
           "note": "paper-only: needs simultaneous positions on 2 venues; overlaps sibling arbitrage thread"}
    # regime table on cumulative basis
    cum = strat_ret.cumsum()
    reg_rows = []
    tmp = pd.DataFrame({"date": gap.index, "pnl": strat_ret}).dropna()
    tmp["regime"] = regime_split(tmp["date"])
    for reg, g in tmp.groupby("regime", sort=True):
        yrs = len(g) * dt_years
        apy = float((1 + g["pnl"]).prod() ** (1 / yrs) - 1) if yrs > 0 else 0.0
        reg_rows.append({"signal": "S4 funding-gap carry", "regime": reg,
                        "apy_net": round(apy, 4), "n_intervals": len(g),
                        "mean_gap_ann": round(float(gap.loc[g.index].abs().mean()), 4)})
    return out, pd.DataFrame(reg_rows)

def s4_split_sample(bmx, hl):
    """Post-hoc diagnostic (NOT the pre-registered test): pick the static
    carry sign on 2023-05..2024-12, evaluate on 2025-01..2026-09.
    Reports APY on notional and on 2x collateral (margin posted on both venues)."""
    hl_a = annualize_funding(hl, 1.0).set_index("ts")["ann"]
    bmx_a = annualize_funding(bmx, 8.0).set_index("ts")["ann"]
    grid = pd.date_range("2023-05-12", "2026-09-16", freq="8h", tz="UTC")
    h = hl_a.reindex(grid, method="ffill", limit=3)
    o = bmx_a.reindex(grid, method="ffill", limit=3)
    gap = (h - o).dropna()
    dt = 8 / (ANN * 24)
    tr = gap[gap.index < "2025-01-01"]
    te = gap[gap.index >= "2025-01-01"]
    sign = 1.0 if tr.mean() > 0 else -1.0
    rows = []
    for tag, g in [("train 2023-05..2024-12", tr), ("test 2025-01..2026-09", te)]:
        pnl = sign * g * dt
        tot = float((1 + pnl).prod() - 1)
        yrs = len(g) * dt
        apy = (1 + tot) ** (1 / yrs) - 1
        rows.append({"split": tag, "sign": sign, "mean_gap_ann": round(float(g.mean()), 4),
                     "apy_notionall": round(apy, 4),
                     "apy_2x_collateral": round(apy / 2, 4),
                     "tbill_benchmark": 0.045})
    return pd.DataFrame(rows)

def main():
    sc, cm, btc, bmx, hl = load()
    print(f"coverage: sc {sc['date'].min()}->{sc['date'].max()} | cm {cm['date'].min()}->{cm['date'].max()} | "
          f"btc {btc['date'].min()}->{btc['date'].max()} | bmx {bmx['ts'].min()}->{bmx['ts'].max()} | "
          f"hl {hl['ts'].min()}->{hl['ts'].max()}")
    results = {}
    r1, t1 = test_s1(sc, btc); results["S1"] = r1
    r2, t2 = test_s2(sc, cm, btc); results["S2"] = r2
    r3, t3 = test_s3(cm, btc); results["S3"] = r3
    r4, t4 = test_s4(bmx, hl, btc); results["S4"] = r4
    t4split = s4_split_sample(bmx, hl)
    print("\n--- S4 split-sample diagnostic (post-hoc) ---")
    print(t4split.to_string(index=False))
    for k, v in results.items():
        print(f"\n=== {k} ===")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")
    print("\n--- regime tables ---")
    for t in (t1, t2, t3):
        print(t.to_string(index=False)); print()
    print(t4.to_string(index=False))
    import json
    with open(os.path.expanduser("~/workspace/trading-bot/results/onchain_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    pd.concat([t1, t2, t3]).to_csv(os.path.expanduser("~/workspace/trading-bot/results/onchain_regimes.csv"), index=False)
    t4.to_csv(os.path.expanduser("~/workspace/trading-bot/results/onchain_s4_regimes.csv"), index=False)
    t4split.to_csv(os.path.expanduser("~/workspace/trading-bot/results/onchain_s4_splitsample.csv"), index=False)

if __name__ == "__main__":
    main()
