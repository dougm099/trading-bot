"""Long-horizon ML simulation battery. See results/ml_longterm_sims_RULEBOOK.md
(frozen 2026-09-20 before any variant ran). Reuses the validated pipeline in
scripts/ml_longterm_experiment.py (imported as exp): features, labels,
embargo, walk-forward, costs. Variants differ only in model/horizon/breadth/
side/features/rebalance — one dimension at a time from the validated anchor.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(__file__))

import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import ml_longterm_experiment as exp

HGB3 = HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                      learning_rate=0.05, random_state=0)
HGB5 = HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                                      learning_rate=0.05, random_state=0)
LOGIT1 = LogisticRegression(C=1.0, max_iter=2000)

DATA, RES = exp.DATA, exp.RES
TEST_YEARS = exp.TEST_YEARS
FEATS_EX = [f for f in exp.FEATS if f not in ("mom_12_1", "mom_6", "sector_rel_mom")]
ET = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=50,
                          random_state=0, n_jobs=-1)


def fit_proba(df, idx, horizon, configs, feats):
    """Walk-forward fit for given (name, model) configs on given features."""
    df = df.copy()
    for name, _ in configs:
        df[f"p_{name}"] = np.nan
    for Y in TEST_YEARS:
        test_dates = sorted(d for d in df["date"].unique() if d.year == Y)
        if not test_dates:
            continue
        cutoff = exp.embargo_cutoff(idx, test_dates[0], horizon)
        assert idx[idx.get_loc(cutoff) + horizon + 1] <= test_dates[0]
        train = df[df["date"] <= cutoff]
        test = df[df["date"].isin(test_dates)]
        Xtr, ytr = train[feats].to_numpy(), train["label"].to_numpy()
        Xte = test[feats].to_numpy()
        for name, mdl in configs:
            mdl.fit(Xtr, ytr)
            df.loc[test.index, f"p_{name}"] = mdl.predict_proba(Xte)[:, 1]
    return df


def simulate_long(df, idx, closes, score_col, test_dates, top_n, cost):
    """Long-only monthly rotation, parameterized breadth/cost/dates."""
    test = df.dropna(subset=[score_col]).copy()
    rets = closes.pct_change()
    val, vals = 1.0, {}
    w, names_prev, prev_entry = None, [], None
    cohorts = []
    for k, d in enumerate(test_dates):
        g = test[test["date"] == d].sort_values(score_col, ascending=False).head(top_n)
        entry = idx[idx.get_loc(d) + 1]
        names = g["ticker"].tolist()
        tgt = pd.Series(1.0 / len(names), index=names)
        if w is not None:
            r_prev = (closes.loc[entry, names_prev] / closes.loc[prev_entry, names_prev] - 1).fillna(0)
            drifted = w * (1 + r_prev)
            drifted = drifted / drifted.sum()
            turnover = 0.5 * (tgt.sub(drifted, fill_value=0)).abs().sum()
        else:
            turnover = 1.0
        c = turnover * cost
        val *= (1 - c)
        nxt = idx[idx.get_loc(test_dates[k + 1]) + 1] if k < len(test_dates) - 1 else idx[-1]
        mret = rets.loc[entry:nxt, names].mean(axis=1, skipna=True).fillna(0)
        g_ret = float(np.prod(1 + mret) - 1)
        for day, r in mret.items():
            val *= (1 + r)
            vals[day] = val
        cohorts.append({"date": d, "net_ret": (1 + g_ret) * (1 - c) - 1,
                        "n": len(names), "turnover": float(turnover),
                        "cost": float(c)})
        w, names_prev, prev_entry = tgt, names, entry
    eq = pd.Series(vals).sort_index()
    return eq / eq.iloc[0], pd.DataFrame(cohorts)


def simulate_ls(df, idx, closes, score_col, test_dates, cost, n_each=25):
    """Dollar-neutral: 50% long top-n_each / 50% short bottom-n_each."""
    test = df.dropna(subset=[score_col]).copy()
    rets = closes.pct_change()
    val, vals = 1.0, {}
    wL = wS = None
    nL_prev, nS_prev, prev_entry = [], [], None
    cohorts = []
    for k, d in enumerate(test_dates):
        g = test[test["date"] == d]
        longs = g.sort_values(score_col, ascending=False).head(n_each)["ticker"].tolist()
        shorts = g.sort_values(score_col, ascending=True).head(n_each)["ticker"].tolist()
        entry = idx[idx.get_loc(d) + 1]
        tgtL = pd.Series(0.5 / len(longs), index=longs)
        tgtS = pd.Series(0.5 / len(shorts), index=shorts)
        if wL is not None:
            rL = (closes.loc[entry, nL_prev] / closes.loc[prev_entry, nL_prev] - 1).fillna(0)
            rS = (closes.loc[entry, nS_prev] / closes.loc[prev_entry, nS_prev] - 1).fillna(0)
            dL = (wL * (1 + rL)); dL = dL / dL.sum() * 0.5
            dS = (wS * (1 + rS)); dS = dS / dS.sum() * 0.5
            turnover = 0.5 * (tgtL.sub(dL, fill_value=0)).abs().sum() \
                     + 0.5 * (tgtS.sub(dS, fill_value=0)).abs().sum()
        else:
            turnover = 1.0
        c = turnover * cost
        val *= (1 - c)
        nxt = idx[idx.get_loc(test_dates[k + 1]) + 1] if k < len(test_dates) - 1 else idx[-1]
        rl = rets.loc[entry:nxt, longs].mean(axis=1, skipna=True).fillna(0)
        rs = rets.loc[entry:nxt, shorts].mean(axis=1, skipna=True).fillna(0)
        mret = 0.5 * rl - 0.5 * rs
        g_ret = float(np.prod(1 + mret) - 1)
        for day, r in mret.items():
            val *= (1 + r)
            vals[day] = val
        cohorts.append({"date": d, "net_ret": (1 + g_ret) * (1 - c) - 1,
                        "n": len(longs) + len(shorts), "turnover": float(turnover),
                        "cost": float(c)})
        wL, wS, nL_prev, nS_prev, prev_entry = tgtL, tgtS, longs, shorts, entry
    eq = pd.Series(vals).sort_index()
    return eq / eq.iloc[0], pd.DataFrame(cohorts)


def subperiod_sharpe(equity):
    r = equity.pct_change().dropna()
    out = {}
    for tag, yrs in (("2016_2020", range(2016, 2021)), ("2021_2025", range(2021, 2026))):
        s = r[r.index.year.isin(yrs)]
        out[tag] = float(s.mean() / s.std() * math.sqrt(252)) if len(s) > 50 and s.std() > 0 else float("nan")
    return out


def cohort_excess(strat_coh, bench_eq, test_dates, idx):
    """Strategy cohort net_ret minus matching benchmark cohort return."""
    xs = []
    for _, row in strat_coh.iterrows():
        d = row["date"]
        k = list(test_dates).index(d)
        e0 = idx[idx.get_loc(d) + 1]
        e1 = idx[idx.get_loc(test_dates[k + 1]) + 1] if k < len(test_dates) - 1 else idx[-1]
        bret = bench_eq.loc[e1] / bench_eq.loc[e0] - 1
        xs.append(row["net_ret"] - bret)
    return pd.Series(xs, index=strat_coh["date"])


def main():
    closes, highs, volumes, sectors = exp.load()
    idx = closes.index

    # ---- sample frames: reuse cached where identical, build 126d fresh ----
    s21 = pd.read_parquet(f"{DATA}/samples_21d.parquet")
    s42 = pd.read_parquet(f"{DATA}/samples_42d.parquet")
    s63 = pd.read_parquet(f"{DATA}/samples_63d.parquet")
    for df in (s21, s42, s63):
        assert set(exp.FEATS) <= set(df.columns)
    print("building 126d samples…", flush=True)
    s126, _ = exp.build_samples(closes, highs, volumes, sectors, 126)
    s126.to_parquet(f"{DATA}/samples_126d.parquet")
    print(f"126d samples: {len(s126)}", flush=True)

    # ---- fits (cached parquets carry no proba columns; refit deterministically) ----
    print("fitting hgb_d3 21d + 42d…", flush=True)
    s21 = fit_proba(s21, idx, 21, [("hgb_d3", HGB3)], exp.FEATS)
    s42 = fit_proba(s42, idx, 42, [("hgb_d3", HGB3)], exp.FEATS)
    print("fitting 63d configs (hgb_d3/d5, logit, extra-trees)…", flush=True)
    s63 = fit_proba(s63, idx, 63, [("hgb_d3", HGB3), ("hgb_d5", HGB5),
                                   ("logit_c1", LOGIT1), ("et", ET)], exp.FEATS)
    print("fitting hgb_d3 ex-momentum 63d…", flush=True)
    s63 = fit_proba(s63, idx, 63, [("hgb_d3_exmom", HGB3)], FEATS_EX)
    print("fitting hgb_d3 126d…", flush=True)
    s126 = fit_proba(s126, idx, 126, [("hgb_d3", HGB3)], exp.FEATS)

    frames = {21: s21, 42: s42, 63: s63, 126: s126}
    monthly = {h: sorted(d for d in frames[h]["date"].unique() if d.year in TEST_YEARS)
               for h in frames}
    skip1 = monthly[63][::2]

    beq_m = exp.benchmark(idx, closes, monthly[63])
    beq_s = exp.benchmark(idx, closes, skip1)
    bm_m, bm_s = exp.perf(beq_m), exp.perf(beq_s)
    bm_sub = subperiod_sharpe(beq_m)
    print(f"benchmark monthly: sharpe={bm_m['sharpe']:.2f} cagr={bm_m['cagr']:.3f}", flush=True)

    # ---- 12-config family ----
    grid = [
        # (id, horizon, score_col, side, top_n, dates, feats_note)
        ("H21", 21, "p_hgb_d3", "long", 25, monthly[21], "full"),
        ("H42", 42, "p_hgb_d3", "long", 25, monthly[42], "full"),
        ("H126", 126, "p_hgb_d3", "long", 25, monthly[126], "full"),
        ("M_d5", 63, "p_hgb_d5", "long", 25, monthly[63], "full"),
        ("M_logit", 63, "p_logit_c1", "long", 25, monthly[63], "full"),
        ("M_et", 63, "p_et", "long", 25, monthly[63], "full"),
        ("B10", 63, "p_hgb_d3", "long", 10, monthly[63], "full"),
        ("B50", 63, "p_hgb_d3", "long", 50, monthly[63], "full"),
        ("LS", 63, "p_hgb_d3", "ls", 25, monthly[63], "full"),
        ("F_momonly", 63, "mom_12_1", "long", 25, monthly[63], "mom_12_1 only"),
        ("F_exmom", 63, "p_hgb_d3_exmom", "long", 25, monthly[63], "ex-momentum"),
        ("R_skip1", 63, "p_hgb_d3", "long", 25, skip1, "full"),
    ]
    rows, curves, cohort_map = [], {}, {}
    for cid, h, scol, side, topn, dates, fnote in grid:
        df = frames[h]
        if side == "ls":
            eq, coh = simulate_ls(df, idx, closes, scol, dates, exp.COST)
            bench, bsub = None, None
        else:
            eq, coh = simulate_long(df, idx, closes, scol, dates, topn, exp.COST)
            bench = bm_s if cid == "R_skip1" else bm_m
            bsub = subperiod_sharpe(beq_s) if cid == "R_skip1" else bm_sub
        m = exp.perf(eq)
        sub = subperiod_sharpe(eq)
        # momentum-rank correlation of the score
        if scol.startswith("p_"):
            te = df.dropna(subset=[scol])
            cors = [spearmanr(g[scol], g["mom_12_1"])[0]
                    for _, g in te.groupby("date") if len(g) >= 30]
            momcorr = float(np.nanmean(cors))
        else:
            momcorr = float("nan")
        rows.append({"config": cid, "horizon": h, "model": scol, "side": side,
                     "top_n": topn, "features": fnote,
                     "sharpe": m["sharpe"], "cagr": m["cagr"],
                     "maxdd": m["maxdd"], "n_days": m["n_days"],
                     "skew": m["skew"], "kurt": m["kurt"],
                     "hit_rate": float((coh["net_ret"] > 0).mean()),
                     "n_cohorts": len(coh),
                     "avg_turnover": float(coh["turnover"].mean()),
                     "total_cost": float(coh["cost"].sum()),
                     "sharpe_16_20": sub["2016_2020"], "sharpe_21_25": sub["2021_2025"],
                     "bench_sharpe": bench["sharpe"] if bench else 0.0,
                     "bench_sharpe_16_20": bsub["2016_2020"] if bsub else 0.0,
                     "bench_sharpe_21_25": bsub["2021_2025"] if bsub else 0.0,
                     "mom_rank_corr": momcorr})
        curves[cid] = eq
        cohort_map[cid] = (coh, dates)
        b = rows[-1]["bench_sharpe"]
        print(f"  {cid:10s} sharpe={m['sharpe']:.2f} (bench {b:.2f}) "
              f"cagr={m['cagr']:.3f} maxdd={m['maxdd']:.3f} "
              f"sub={sub['2016_2020']:.2f}/{sub['2021_2025']:.2f} "
              f"momcorr={momcorr:.2f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{RES}/ml_longterm_sims.csv", index=False)

    # ---- family DSR + Holm (K=12, best new config by Sharpe) ----
    trial_daily = np.array([r["sharpe"] / math.sqrt(252) for r in rows])
    best = max(rows, key=lambda r: r["sharpe"])
    dsr = exp.deflated_sharpe(best["sharpe"] / math.sqrt(252), best["n_days"],
                              best["skew"], best["kurt"], trial_daily)
    ps = [float(__import__("scipy.stats", fromlist=["t"]).t.sf(
        r["sharpe"] / math.sqrt(252) * math.sqrt(r["n_days"]), r["n_days"] - 1))
        for r in rows]
    holm = exp.holm_adjust(ps)
    print(f"BEST={best['config']} sharpe={best['sharpe']:.2f}; "
          f"DSR(K=12)={dsr:.3f}", flush=True)

    # ---- stress tests on best 2 by Sharpe ----
    top2 = sorted(rows, key=lambda r: r["sharpe"], reverse=True)[:2]
    stress = {}
    for r in top2:
        cid = r["config"]
        df = frames[r["horizon"]]
        scol = r["model"]
        dates = dict(R_skip1=skip1).get(cid, monthly[r["horizon"]])
        sim = simulate_ls if r["side"] == "ls" else simulate_long
        kw = {} if r["side"] == "ls" else {"top_n": r["top_n"]}
        out = {"config": cid}
        for bps in (0.0005, 0.0025):
            eq_c, _ = sim(df, idx, closes, scol, dates, cost=bps, **kw)
            out[f"sharpe_{int(bps*10000)}bps"] = exp.perf(eq_c)["sharpe"]
            out[f"cagr_{int(bps*10000)}bps"] = exp.perf(eq_c)["cagr"]
        eq = curves[cid]
        beq = beq_s if cid == "R_skip1" else beq_m
        out["mar2020"] = float(eq.loc["2020-03"].iloc[-1] / eq.loc["2020-03"].iloc[0] - 1)
        out["mar2020_bench"] = float(beq.loc["2020-03"].iloc[-1] / beq.loc["2020-03"].iloc[0] - 1)
        out["y2022"] = float(eq.loc["2022"].iloc[-1] / eq.loc["2022"].iloc[0] - 1)
        out["y2022_bench"] = float(beq.loc["2022"].iloc[-1] / beq.loc["2022"].iloc[0] - 1)
        rs, rb = eq.pct_change().dropna(), beq.pct_change().dropna()
        al = rs.align(rb, join="inner")
        out["beta_vs_bench"] = float(np.polyfit(al[1], al[0], 1)[0])
        coh, cdates = cohort_map[cid]
        xs = cohort_excess(coh, beq, cdates, idx)
        tot = xs.sum()
        out["top5m_share_excess"] = float(xs.nlargest(5).sum() / tot) if tot != 0 else float("nan")
        out["worst_cohort"] = float(xs.min())
        stress[cid] = out
        print(f"  stress {cid}: " +
              " ".join(f"{k}={v:.3f}" for k, v in out.items() if k != "config"),
              flush=True)

    import json
    with open(f"{RES}/ml_longterm_sims_summary.json", "w") as f:
        json.dump({"dsr_K12": dsr, "best": best["config"],
                   "holm": {r["config"]: float(p) for r, p in zip(rows, holm)},
                   "benchmark": {"sharpe": bm_m["sharpe"], "cagr": bm_m["cagr"],
                                 "maxdd": bm_m["maxdd"]},
                   "stress": stress}, f, indent=1, default=float)
    print("done", flush=True)


if __name__ == "__main__":
    main()
