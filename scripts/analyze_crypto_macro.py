"""Crypto<->equities bridge + macro overlay research for the trading bot.

Reads cached raw data (see __main__), runs the research threads from the
task spec with strict no-lookahead alignment, and backtests every candidate
with real numbers (cost-aware). Prints key tables and saves CSVs to results/.

Signal convention: for a trade executed at time T, the signal uses only
data with timestamps < T. Anchors below are documented explicitly.

Author: research subagent, 2026-09-20.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, pearsonr

RESULTS = os.path.expanduser("~/workspace/trading-bot/results")
CACHE = os.path.join(RESULTS, "cache")
ET = "America/New_York"

# Costs: 10 bps/side crypto, 5 bps/side equities
COST_CRYPTO = 0.0010
COST_EQUITY = 0.0005


# ---------------------------------------------------------------- helpers
def granger_pvalues(y: pd.Series, x: pd.Series, max_lag: int = 5) -> dict:
    """Granger causality: does x Granger-cause y? OLS F-test, no lookahead
    (only lags). Returns {lag: p-value}."""
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    yv, xv = d["y"].to_numpy(), d["x"].to_numpy()
    n = len(yv)
    out = {}
    for lag in range(1, max_lag + 1):
        if n - lag < 4 * lag + 2:
            out[lag] = np.nan
            continue
        Y = yv[lag:]
        Xu = np.column_stack([np.ones(n - lag)] +
                             [yv[lag - k:-k if k else None] for k in range(1, lag + 1)] +
                             [xv[lag - k:-k if k else None] for k in range(1, lag + 1)])
        Xr = np.column_stack([np.ones(n - lag)] +
                             [yv[lag - k:-k if k else None] for k in range(1, lag + 1)])
        bu, *_ = np.linalg.lstsq(Xu, Y, rcond=None)
        br, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
        ssr_u = float(((Y - Xu @ bu) ** 2).sum())
        ssr_r = float(((Y - Xr @ br) ** 2).sum())
        q = lag
        df1 = Xu.shape[0] - Xu.shape[1]
        if ssr_u <= 0 or df1 <= 0:
            out[lag] = np.nan
            continue
        F = ((ssr_r - ssr_u) / q) / (ssr_u / df1)
        out[lag] = float(f_dist.sf(max(F, 0), q, df1))
    return out


def corr_stats(a: pd.Series, b: pd.Series):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 30:
        return np.nan, np.nan, 0
    r, p = pearsonr(d["a"], d["b"])
    return float(r), float(p), len(d)


def lagged_corr(y: pd.Series, x: pd.Series, lags=(-24, 24)):
    """corr(y_t, x_{t-lag}) for each lag; both series must already be aligned
    on a common index (NaNs dropped pairwise)."""
    lo, hi = lags
    out = {}
    for lag in range(lo, hi + 1):
        r, p, n = corr_stats(y, x.shift(lag))
        out[lag] = (r, p, n)
    return out


def summarize_bt(rets: pd.Series, label: str) -> dict:
    """rets: per-trade net returns (after costs)."""
    rets = rets.dropna()
    n = len(rets)
    if n == 0:
        return {"strategy": label, "trades": 0}
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    # crude daily-ish annualization: use per-trade Sharpe scaled by sqrt(n per year)
    days = max((rets.index.max() - rets.index.min()).days, 1) if isinstance(rets.index, pd.DatetimeIndex) else n
    per_year = n / days * 365.25
    sharpe = (rets.mean() / rets.std() * np.sqrt(per_year)) if rets.std() > 0 else 0.0
    dd = (cum / cum.cummax() - 1).min()
    win = (rets > 0).mean()
    return {"strategy": label, "trades": n, "net_return": round(float(total), 4),
            "ann_sharpe": round(float(sharpe), 3), "max_dd": round(float(dd), 4),
            "win_rate": round(float(win), 3),
            "mean_per_trade_bps": round(float(rets.mean() * 1e4), 1)}


# ------------------------------------------------------- data loading
def load_data():
    """Load cached raw data; returns dict of DataFrames."""
    data = {}
    for sym in ("BTC", "ETH", "SOL"):
        df = pd.read_pickle(f"{CACHE}/crypto_{sym}.pkl")
        df.index = pd.to_datetime(df.index, utc=True)
        data[sym] = df.sort_index()
    eq = pd.read_pickle(f"{CACHE}/equity_hourly.pkl")
    if isinstance(eq.columns, pd.MultiIndex):
        eq.columns = ["_".join(c).strip() for c in eq.columns.values]
    data["eq_h"] = eq
    daily = pd.read_pickle(f"{CACHE}/daily.pkl")
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = ["_".join(c).strip() for c in daily.columns.values]
    data["daily"] = daily
    return data


def build_sessions(eq_h: pd.DataFrame, tickers=("SPY", "QQQ", "COIN", "MSTR", "MARA")):
    """Per-trading-day session OHLC from hourly bars (America/New_York).
    Returns dict ticker -> DataFrame(date -> open, close, gap, first_hour_ret)."""
    out = {}
    for t in tickers:
        sub = eq_h[[f"Open_{t}", f"Close_{t}"]].dropna()
        sub = sub.tz_convert(ET)
        rows = []
        for date, g in sub.groupby(sub.index.date):
            g = g.sort_index()
            if len(g) < 6:
                continue  # require ~full session
            o = g[f"Open_{t}"].iloc[0]
            c = g[f"Close_{t}"].iloc[-1]
            sess_ret = c / o - 1
            first_hour = g[f"Close_{t}"].iloc[0] / o - 1  # 9:30->10:30 bar
            rows.append((date, o, c, sess_ret, first_hour))
        df = pd.DataFrame(rows, columns=["date", "open", "close", "sess_ret", "first_hour_ret"])
        df["date"] = pd.to_datetime(df["date"])
        out[t] = df.set_index("date")
    return out


def crypto_anchors(crypto: pd.DataFrame):
    """Hourly crypto close reindexed to ET; returns price getters by ET bar label.

    Conventions (all no-lookahead):
      - px_16  : close of bar labeled 15:00 ET  (= price at 16:00 ET, equity close)
      - px_20  : close of bar labeled 19:00 ET  (= price at 20:00 ET)
      - px_24  : close of bar labeled 23:00 ET  (= price at 24:00 ET)
      - px_09  : close of bar labeled 08:00 ET  (= price at 09:00 ET, ~30min before equity open)
    overnight = px_09(next day) / px_16(day) - 1
    evening   = px_20(day) / px_16(day) - 1   (tradeable: enters at 16:00 ET close)
    late      = px_24(day) / px_16(day) - 1
    """
    px = crypto["close"].tz_convert(ET).resample("1h").last().dropna()
    idx = pd.DataFrame({"px": px})
    idx["hour"] = idx.index.hour
    by = {}
    for h, name in [(15, "px_16"), (19, "px_20"), (23, "px_24"), (8, "px_09")]:
        s = idx.loc[idx["hour"] == h, "px"].copy()
        s.index = s.index.normalize().tz_localize(None)
        by[name] = s.rename(name)
    return by


def thread_1a(session_ret: pd.Series, gap_ret: pd.Series, first_hour: pd.Series,
              anchors: dict, vix_regime: pd.Series | None = None):
    """Equity session <-> crypto overnight lead-lag."""
    px16, px09, px20 = anchors["px_16"], anchors["px_09"], anchors["px_20"]
    overnight = px09 / px16.shift(1) - 1          # day D overnight uses BTC 16:00 D-1 -> 09:00 D
    evening = px20 / px16 - 1                     # 16:00 -> 20:00 ET same day D
    df = pd.DataFrame({"sess": session_ret, "overnight": overnight,
                       "evening": evening, "gap": gap_ret, "fh": first_hour}).dropna()
    res = {"n": len(df)}
    # direction 1: equity session -> crypto overnight (next obs overnight is D+1's, aligned here)
    df["overnight_next"] = df["overnight"].shift(-1)
    res["corr_sess->overnight_next"] = corr_stats(df["sess"], df["overnight_next"])
    res["corr_sess->evening_same_day"] = corr_stats(df["sess"], df["evening"])
    # direction 2: crypto overnight -> equity next day
    res["corr_overnight->gap"] = corr_stats(df["overnight"], df["gap"])
    res["corr_overnight->first_hour"] = corr_stats(df["overnight"], df["fh"])
    # Granger both directions on daily obs
    d2 = df[["sess", "overnight"]].dropna()
    res["granger_sess->overnight"] = granger_pvalues(d2["overnight"], d2["sess"])
    res["granger_overnight->sess"] = granger_pvalues(d2["sess"], d2["overnight"])
    if vix_regime is not None:
        hi = df[vix_regime.reindex(df.index).fillna(False)]
        lo = df[~vix_regime.reindex(df.index).fillna(False)]
        res["regime_corr_sess->overnight_next_high_vix"] = corr_stats(hi["sess"], hi["overnight_next"])
        res["regime_corr_sess->overnight_next_low_vix"] = corr_stats(lo["sess"], lo["overnight_next"])
        res["regime_corr_overnight->gap_high_vix"] = corr_stats(hi["overnight"], hi["gap"])
        res["regime_corr_overnight->gap_low_vix"] = corr_stats(lo["overnight"], lo["gap"])
        res["n_high_vix"] = len(hi)
    return res, df


def thread_1b(sessions: dict, anchors: dict):
    """BTC overnight -> COIN/MSTR/MARA open & first hour; equity session -> BTC evening."""
    px16, px09, px20 = anchors["px_16"], anchors["px_09"], anchors["px_20"]
    overnight = px09 / px16.shift(1) - 1
    evening = px20 / px16 - 1
    out = {}
    for t in ("COIN", "MSTR", "MARA"):
        s = sessions[t]
        df = pd.DataFrame({"sess": s["sess_ret"], "gap": s["open"] / s["close"].shift(1) - 1,
                           "fh": s["first_hour_ret"], "overnight": overnight, "evening": evening})
        df = df.dropna()
        r = {"n": len(df)}
        r["corr_overnight_btc->open_gap"] = corr_stats(df["overnight"], df["gap"])
        r["corr_overnight_btc->first_hour"] = corr_stats(df["overnight"], df["fh"])
        r["corr_sess->btc_evening"] = corr_stats(df["sess"], df["evening"])
        r["beta_overnight->first_hour"] = float(np.polyfit(df["overnight"], df["fh"], 1)[0]) if len(df) > 30 else np.nan
        out[t] = (r, df)
    return out


# (hourly cross-correlation is computed inline in main with gap masking)


def thread_1c(data):
    """ETH/BTC ratio dynamics: autocorrelation of ratio changes, daily momentum persistence."""
    ratio = data["ETH"]["close"] / data["BTC"]["close"]
    lr = np.log(ratio)
    dret = lr.resample("1d").last().diff()          # daily change of log ratio
    out = {"n_daily": int(dret.count())}
    ac = {}
    for lag in (1, 2, 5, 7, 14, 30):
        r, p, n = corr_stats(dret, dret.shift(lag))
        ac[lag] = (round(r, 4), round(p, 4) if not np.isnan(p) else None, n)
    out["autocorr_daily_change"] = ac
    # momentum: past-20d ratio return -> next-5d ratio return
    mom = lr.resample("1d").last()
    past20 = mom / mom.shift(20) - 1
    fwd5 = mom.shift(-5) / mom - 1
    r, p, n = corr_stats(past20, fwd5)
    out["mom_20d->fwd_5d"] = (round(r, 4), round(p, 4) if p == p else None, n)
    # hit rate of sign persistence
    dd = pd.DataFrame({"p": past20, "f": fwd5}).dropna()
    out["sign_hit_rate"] = round(float((np.sign(dd["p"]) == np.sign(dd["f"])).mean()), 4)
    out["n_mom"] = len(dd)
    return out, dret


def thread_2(data, daily_close, vix_high: pd.Series):
    """Macro overlays on daily bars. daily_close: ticker->close series (2-5y sample)."""
    out = {}
    btc = data["BTC"]["close"].tz_convert("UTC").resample("1d").last().pct_change()
    btc.index = btc.index.normalize().tz_localize(None)
    get = lambda t: daily_close[t]
    rets = {t: get(t).pct_change() for t in ("GLD", "UUP", "TLT", "^VIX", "SPY", "QQQ")}
    rets["BTC"] = btc
    R = pd.DataFrame(rets)

    # 2a DXY proxy vs BTC
    r, p, n = corr_stats(R["UUP"], R["BTC"])
    Rm = R.resample("M").apply(lambda s: (1 + s).prod() - 1)
    rm, pm, nm = corr_stats(Rm["UUP"], Rm["BTC"])
    out["2a_uup_btc_daily_corr"] = (round(r, 4), round(p, 4), n)
    out["2a_uup_btc_monthly_corr"] = (round(rm, 4), round(pm, 4), nm)

    # 2b TLT vs QQQ
    r, p, n = corr_stats(R["TLT"], R["QQQ"])
    Rw = R.resample("W").apply(lambda s: (1 + s).prod() - 1)
    rw, pw, nw = corr_stats(Rw["TLT"], Rw["QQQ"])
    out["2b_tlt_qqq_daily_corr"] = (round(r, 4), round(p, 4), n)
    out["2b_tlt_qqq_weekly_corr"] = (round(rw, 4), round(pw, 4), nw)

    # 2c GLD vs BTC: returns, drawdown, beta
    r, p, n = corr_stats(R["GLD"], R["BTC"])
    out["2c_gld_btc_daily_corr"] = (round(r, 4), round(p, 4), n)
    px = pd.DataFrame({"BTC": (1 + R["BTC"]).cumprod(), "GLD": (1 + R["GLD"]).cumprod()}).dropna()
    dd_btc = px["BTC"] / px["BTC"].cummax() - 1
    dd_gld = px["GLD"] / px["GLD"].cummax() - 1
    rdd, pdd, ndd = corr_stats(dd_btc, dd_gld)
    out["2c_drawdown_level_corr"] = (round(rdd, 4), round(pdd, 4), ndd)
    d2 = pd.DataFrame({"BTC": R["BTC"], "GLD": R["GLD"]}).dropna()
    beta = float(np.polyfit(d2["GLD"], d2["BTC"], 1)[0])
    out["2c_btc_beta_to_gold"] = round(beta, 3)
    out["2c_n"] = n

    # 2d VIX regime: recompute key lead-lag corrs in high-VIX quartile
    out["2d_vix_high_share"] = round(float(vix_high.mean()), 3)
    return out, R


# ----------------------------------------------------------- backtests
def bt_coin_overnight(df: pd.DataFrame, sessions_t: pd.DataFrame, min_abs: float = 0.0,
                      label="BT1: BTC overnight -> COIN open-close"):
    """Enter at equity open (signal = BTC overnight move, known 09:00 ET),
    exit at equity close. Direction = sign(signal)."""
    sig = df["overnight"].dropna()
    trade = sig.abs() >= min_abs
    idx = sig.index[trade]
    pos = np.sign(sig.loc[idx]).to_numpy()
    px_o = sessions_t.loc[idx, "open"].to_numpy()
    px_c = sessions_t.loc[idx, "close"].to_numpy()
    raw = pos * (px_c / px_o - 1)
    net = pd.Series(raw - 2 * COST_EQUITY, index=pd.to_datetime(idx))
    return summarize_bt(net, label)


def bt_btc_evening(df: pd.DataFrame, spy_sess: pd.Series, anchors: dict,
                   exit_key: str, label: str):
    """Signal = sign(SPY session return), known at 16:00 ET. Long/short BTC
    from 16:00 ET to exit anchor (px_20=20:00 ET or px_09 next day)."""
    px16, pxX = anchors["px_16"], anchors[exit_key]
    if exit_key == "px_09":
        fwd = pxX / px16.shift(1) - 1   # aligned: index D -> 16:00 D-1 .. 09:00 D; we need 16:00 D -> 09:00 D+1
        fwd = fwd.shift(-1)
    else:
        fwd = pxX / px16 - 1
    d = pd.DataFrame({"sig": np.sign(spy_sess), "fwd": fwd}).dropna()
    net = d["sig"] * d["fwd"] - 2 * COST_CRYPTO
    net.index = pd.to_datetime(d.index)
    return summarize_bt(net, label)


def bt_spy_first_hour(df: pd.DataFrame, label="BT3: BTC overnight -> SPY first hour"):
    """Signal = sign(BTC overnight), known 09:00 ET. Enter SPY at open,
    exit at 10:30 ET."""
    net = np.sign(df["overnight"]) * df["fh"] - 2 * COST_EQUITY
    net.index = pd.to_datetime(df.index)
    return summarize_bt(net, label)


def bt_ratio_momentum(data, label="BT4: ETH/BTC 20d ratio momentum (5d hold)"):
    """Signal = sign(past-20d ETH/BTC ratio return), known at day-D close.
    Spread: long ETH / short BTC equal notional. Position only changes when the
    signal flips; each flip costs 4 x 10bps (close 2 legs + open 2 legs)."""
    lr = np.log(data["ETH"]["close"] / data["BTC"]["close"]).resample("1d").last()
    sig = np.sign(lr / lr.shift(20) - 1)
    eret = data["ETH"]["close"].resample("1d").last().pct_change()
    bret = data["BTC"]["close"].resample("1d").last().pct_change()
    d = pd.DataFrame({"sig": sig, "spread": eret - bret}).dropna()
    pos = d["sig"].shift(1)  # trade at next day's prices; signal known at close
    gross = pos * d["spread"]
    flips = (pos.diff().abs() > 0)
    net = gross - flips * 4 * COST_CRYPTO
    net.index = pd.to_datetime(d.index)
    return summarize_bt(net.dropna(), label)


# ---------------------------------------------------------------- main
def main():
    print("Loading cached data...")
    data = load_data()
    sessions = build_sessions(data["eq_h"])
    daily = data["daily"]
    dclose = {t: daily[f"Close_{t}"].dropna() for t in
              ("COIN", "MSTR", "MARA", "SPY", "QQQ", "GLD", "UUP", "TLT", "^VIX", "BTC-USD")}
    vix = dclose["^VIX"]
    vix_high = (vix >= vix.quantile(0.75)).rename("vix_high")

    anchors_btc = crypto_anchors(data["BTC"])

    corr_rows, granger_rows, bt_results = [], [], []

    def add_corr(thread, pair, r, p, n, note=""):
        corr_rows.append({"thread": thread, "pair": pair, "corr": round(r, 4) if r == r else None,
                          "p": round(p, 4) if p == p else None, "n": n, "note": note})

    # ---------------- Thread 1a: SPY & QQQ <-> BTC
    for t in ("SPY", "QQQ"):
        s = sessions[t]
        res, df = thread_1a(s["sess_ret"], s["open"] / s["close"].shift(1) - 1,
                            s["first_hour_ret"], anchors_btc, vix_high)
        print(f"\n=== Thread 1a: {t} <-> BTC (n={res['n']}) ===")
        for k in ("corr_sess->overnight_next", "corr_sess->evening_same_day",
                  "corr_overnight->gap", "corr_overnight->first_hour"):
            r, p, n = res[k]
            print(f"  {k:38s} r={r:+.4f} p={p:.4g} n={n}")
            add_corr("1a", f"{t} {k}", r, p, n)
        for dname, g in (("sess->overnight", res["granger_sess->overnight"]),
                         ("overnight->sess", res["granger_overnight->sess"])):
            best = min(g.items(), key=lambda kv: kv[1] if kv[1] == kv[1] else 99)
            print(f"  granger {dname}: min p={best[1]:.4g} at lag {best[0]}  { {k: round(v,4) for k,v in g.items()} }")
            for lag, pv in g.items():
                granger_rows.append({"thread": "1a", "direction": f"{t} {dname}", "lag": lag,
                                     "p": round(pv, 4) if pv == pv else None})
        r, p, n = res["regime_corr_sess->overnight_next_high_vix"]
        r2, p2, n2 = res["regime_corr_sess->overnight_next_low_vix"]
        print(f"  VIX top-quartile (n={res['n_high_vix']}): sess->overnight_next r={r:+.4f} vs rest r={r2:+.4f}")
        r, p, n = res["regime_corr_overnight->gap_high_vix"]
        r2, p2, n2 = res["regime_corr_overnight->gap_low_vix"]
        print(f"  VIX top-quartile: overnight->gap r={r:+.4f} vs rest r={r2:+.4f}")
        add_corr("1a", f"{t} sess->overnight_next | high VIX", *res["regime_corr_sess->overnight_next_high_vix"][:2],
                 res["regime_corr_sess->overnight_next_high_vix"][2], note=f"n={res['n_high_vix']}")
    spy_df = thread_1a(sessions["SPY"]["sess_ret"], sessions["SPY"]["open"] / sessions["SPY"]["close"].shift(1) - 1,
                       sessions["SPY"]["first_hour_ret"], anchors_btc)[1]

    # ---------------- Thread 1b: BTC <-> crypto equities
    print("\n=== Thread 1b: BTC <-> crypto equities ===")
    t1b = thread_1b(sessions, anchors_btc)
    df_1b = {}
    for t, (r, df) in t1b.items():
        df_1b[t] = df
        for k in ("corr_overnight_btc->open_gap", "corr_overnight_btc->first_hour", "corr_sess->btc_evening"):
            rr, pp, nn = r[k]
            print(f"  {t:5s} {k:36s} r={rr:+.4f} p={pp:.4g} n={nn}")
            add_corr("1b", f"{t} {k}", rr, pp, nn)
        print(f"  {t:5s} beta(overnight->first_hour) = {r['beta_overnight->first_hour']:.3f}")

    # ---------------- hourly cross-correlation on overlapping bars
    print("\n=== Hourly cross-corr (overlap only; lag>0 = equity leads crypto) ===")
    for sym in ("BTC", "ETH", "SOL"):
        data[sym]["hret"] = data[sym]["close"].pct_change()
    # mask equity returns that span a bar gap (>90 min) to avoid weekend-spanning artefacts
    for sym in ("BTC", "ETH", "SOL"):
        c = data[sym]["hret"]
        for t in ("SPY", "QQQ"):
            e = data["eq_h"][f"Close_{t}"].tz_convert("UTC").pct_change()
            # equity bars are labeled :30 (9:30 ET etc); floor to the hour so they
            # align with crypto's on-the-hour bars (30-min label offset, same market interval)
            e.index = e.index.floor("h")
            dt = e.index.to_series().diff().dt.total_seconds()
            e = e.mask(dt > 5400)
            common = pd.DataFrame({"c": c, "e": e}).dropna()
            peak = {}
            for lag in range(-12, 13):
                r, p, n = corr_stats(common["c"], common["e"].shift(lag))
                peak[lag] = r
            pk = max(peak.items(), key=lambda kv: abs(kv[1]) if kv[1] == kv[1] else -1)
            print(f"  {sym}/{t}: n={len(common)}, peak |r| at lag {pk[0]}: r={pk[1]:+.4f}; lag0 r={peak[0]:+.4f}, lag+1 r={peak[1]:+.4f}")
            add_corr("1a-hourly", f"{sym}/{t} peak-lag", pk[1], np.nan, len(common), note=f"lag={pk[0]}")
            add_corr("1a-hourly", f"{sym}/{t} lag0", peak[0], np.nan, len(common))

    # ---------------- Thread 1c: ETH/BTC ratio
    print("\n=== Thread 1c: ETH/BTC ratio ===")
    r1c, _ = thread_1c(data)
    print(f"  daily change autocorr: {r1c['autocorr_daily_change']}")
    r, p, n = r1c["mom_20d->fwd_5d"]
    print(f"  20d ratio momentum -> fwd 5d: r={r:+.4f} p={p} n={n}; sign hit rate={r1c['sign_hit_rate']}")

    # ---------------- Thread 2: macro overlays
    print("\n=== Thread 2: macro overlays (daily) ===")
    r2, R = thread_2(data, dclose, vix_high)
    for k, v in r2.items():
        print(f"  {k}: {v}")
        if isinstance(v, tuple) and len(v) == 3:
            add_corr("2", k, v[0], v[1], v[2])

    # ---------------- Backtests (cost-aware)
    print("\n=== Cost-aware backtests ===")
    bt_results.append(bt_coin_overnight(df_1b["COIN"], sessions["COIN"]))
    bt_results.append(bt_coin_overnight(df_1b["COIN"], sessions["COIN"], min_abs=0.005,
                                        label="BT1b: BTC overnight -> COIN (only |sig|>=0.5%)"))
    bt_results.append(bt_coin_overnight(df_1b["MSTR"], sessions["MSTR"],
                                        label="BT1c: BTC overnight -> MSTR open-close"))
    bt_results.append(bt_coin_overnight(df_1b["MARA"], sessions["MARA"],
                                        label="BT1d: BTC overnight -> MARA open-close"))
    bt_results.append(bt_btc_evening(spy_df, sessions["SPY"]["sess_ret"], anchors_btc, "px_20",
                                     "BT2a: SPY session -> BTC 16:00-20:00 ET"))
    bt_results.append(bt_btc_evening(spy_df, sessions["SPY"]["sess_ret"], anchors_btc, "px_09",
                                     "BT2b: SPY session -> BTC 16:00-09:00 next day"))
    bt_results.append(bt_spy_first_hour(spy_df))
    bt_results.append(bt_ratio_momentum(data))
    for b in bt_results:
        print(f"  {b['strategy']}: trades={b['trades']} net={b.get('net_return')} "
              f"sharpe={b.get('ann_sharpe')} maxDD={b.get('max_dd')} win={b.get('win_rate')} "
              f"mean/trade={b.get('mean_per_trade_bps')}bps")

    # stability check: split-sample (first vs second half) for the two
    # positive-looking candidates — guards against regime-concentrated luck
    print("\n=== Split-sample stability (BT1b COIN, BT1c MSTR) ===")
    for t, filt in (("COIN", 0.005), ("MSTR", 0.0)):
        df, s = df_1b[t], sessions[t]
        sig = df["overnight"].dropna()
        idx = sig.index[sig.abs() >= filt]
        pos = np.sign(sig.loc[idx]).to_numpy()
        raw = pos * (s.loc[idx, "close"].to_numpy() / s.loc[idx, "open"].to_numpy() - 1)
        net = pd.Series(raw - 2 * COST_EQUITY, index=pd.to_datetime(idx))
        mid = net.index[len(net) // 2]
        for name, sub in (("first-half", net[net.index < mid]), ("second-half", net[net.index >= mid])):
            cum = (1 + sub).cumprod()
            print(f"  {t} {name}: n={len(sub)} net={cum.iloc[-1] - 1:+.4f} "
                  f"maxDD={(cum / cum.cummax() - 1).min():.4f} win={(sub > 0).mean():.3f}")

    # ---------------- save CSVs
    pd.DataFrame(corr_rows).to_csv(f"{RESULTS}/crypto_macro_correlations.csv", index=False)
    pd.DataFrame(granger_rows).to_csv(f"{RESULTS}/crypto_macro_granger.csv", index=False)
    pd.DataFrame(bt_results).to_csv(f"{RESULTS}/crypto_macro_backtests.csv", index=False)
    print(f"\nSaved CSVs to {RESULTS}/")


if __name__ == "__main__":
    main()
