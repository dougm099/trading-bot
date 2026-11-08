"""
THREAD B: Trump action event studies.
Methodology:
- Daily data 2017-01-01..2026-09-20: SPY, XLE, USO, UUP (returns), ^VIX (pct change), ^TNX (change in pp).
- For each event date, event trading day t0 = first trading day >= announcement date.
- Estimation window [-65,-6] trading days (60 days).
- Market model: R_asset = alpha + beta * R_SPY + eps for XLE, USO, UUP.
  SPY itself, VIX, TNX: mean-adjusted (own trailing mean/vol from estimation window).
- Abnormal returns over [-5,+10]; report CAR[-1,+1], CAR[+2,+10].
- Event-level t: CAR / (sigma_est * sqrt(k)) with estimation-window residual vol.
- Type aggregation: mean CAR, cross-sectional SE = sd/sqrt(n), t_cs, fraction |t|>2, mean |CAR|.
Read-only analysis. No trading, no keys.
"""
import pickle
import numpy as np
import pandas as pd

# (event_date, label, type, notes/source)
EVENTS = [
    # --- TARIFFS ---
    ("2018-03-01", "Sec 232 steel 25% / aluminum 10% announced", "tariff",
     "Trump announcement; effective Mar 23. Congress CRS R45529."),
    ("2018-07-06", "Sec 301 List 1 ($34B China) 25% imposed", "tariff",
     "Effective date; announced weeks prior."),
    ("2018-09-17", "Sec 301 List 3 ($200B China) 10% announced (eff. Sep 24)", "tariff",
     "Trump announcement Monday Sep 17."),
    ("2019-05-05", "Tweet: List 3 raised 10%->25% (eff. May 10)", "tariff",
     "Sunday tweet -> Mon May 6 trading. (CONFOUND: weekend gap)"),
    ("2019-08-01", "List 4 ($300B) 10% announced (eff. Sep 1)", "tariff",
     "Thursday announcement."),
    ("2019-08-23", "China retaliates; 'hereby ordered' tweet; List 4A ->15%", "tariff",
     "Friday. CONFOUND: same-day Powell 'bigger enemy' tweet (FED)."),
    ("2020-01-15", "Phase One deal signed (US-China)", "tariff",
     "Resolution/positive event. Wednesday."),
    ("2025-02-01", "IEEPA EO: 25% Canada/Mexico, 10% China (eff. Feb 4)", "tariff",
     "Saturday EO -> Mon Feb 3 trading."),
    ("2025-03-04", "Canada/Mexico 25% take effect; China doubled to 20%", "tariff",
     "Trump confirmed Mar 3 'take effect tomorrow'."),
    ("2025-04-02", "Liberation Day: reciprocal tariffs announced", "tariff",
     "Rose Garden, Wed Apr 2 ~4pm ET. After close of trading."),
    ("2025-04-09", "90-day reciprocal pause; China ->125%/145%", "tariff",
     "Wed Apr 9, midday. Positive for non-China."),
    ("2018-12-01", "G20 Buenos Aires: US-China 90-day trade truce", "tariff",
     "POST-HOC robustness add. Saturday -> Mon Dec 3 trading. Positive/resolution."),
    ("2025-06-21", "US strikes on Iran nuclear sites (Operation Midnight Hammer)", "sanction",
     "POST-HOC robustness add. Saturday evening -> Mon Jun 23 trading. Military, not sanctions; oil-relevant."),
    ("2025-08-01", "Reciprocal tariff deadline reinstated (Aug 1)", "tariff",
     "Friday."),
    # --- SANCTIONS ---
    ("2018-05-08", "JCPOA withdrawal announced (Iran sanctions to return)", "sanction",
     "NSPM signed May 8. Tuesday."),
    ("2019-01-28", "PDVSA oil sanctions (EO 13850)", "sanction",
     "White House briefing Monday Jan 28."),
    ("2019-04-22", "End of Iran oil waivers announced (expire May 2)", "sanction",
     "Pompeo announcement Monday Apr 22."),
    # --- ENERGY / SPR / OPEC jawboning ---
    ("2018-04-20", "Tweet: OPEC 'artificially Very High'", "energy",
     "Friday tweet. First OPEC mention of term."),
    ("2018-07-04", "Tweet: OPEC 'REDUCE PRICING NOW!'", "energy",
     "July 4 holiday -> Thu Jul 5 trading. Precision loss noted."),
    ("2019-09-15", "SPR release authorized after Saudi Aramco attack", "energy",
     "Sunday -> Mon Sep 16 trading."),
    ("2020-04-02", "Trump announces talks w/ MBS+Putin on production cuts", "energy",
     "Thursday. OPEC+ 9.7mb/d deal followed Apr 9-12."),
    ("2025-01-23", "Davos: 'demand OPEC bring down oil prices'", "energy",
     "Video address to Davos, Thu Jan 23."),
    # --- FED PRESSURE ---
    ("2018-07-19", "Trump 'not thrilled' with Powell rate hikes", "fed",
     "CNBC interview Thursday Jul 19."),
    ("2018-12-22", "Reports: Trump discussed firing Powell", "fed",
     "Bloomberg Sat Dec 22 -> Mon Dec 24 (half day). News-date precision loss."),
    ("2019-08-23", "Tweet: 'bigger enemy, Jay Powell or Chairman Xi?'", "fed",
     "Friday. CONFOUND: same-day tariff escalation (T6). EXCLUDED from fed agg."),
    ("2025-04-17", "Truth Social: 'Powell's termination cannot come fast enough!'", "fed",
     "Thursday Apr 17."),
    ("2025-04-22", "Backpedal: 'no intention of firing him'", "fed",
     "Tuesday Apr 22. Opposite-signed event."),
    ("2026-04-15", "Fox Business: fire Powell if not resigned by May 15", "fed",
     "Wed Apr 15 interview; Warsh named successor."),
]

TICKERS = ["SPY", "XLE", "USO", "UUP", "^VIX", "^TNX"]
BETA_ASSETS = ["XLE", "USO", "UUP"]
EST_A, EST_B = -65, -6          # estimation window (trading days rel. t0)
WIN_LO, WIN_HI = -5, 10

def load_returns():
    with open("results/raw/trump_events_prices.pkl", "rb") as f:
        px = pickle.load(f)["Close"]
    px = px.dropna()
    dates = px.index
    rets = pd.DataFrame(index=dates)
    for t in ["SPY", "XLE", "USO", "UUP"]:
        rets[t] = px[t].pct_change()
    rets["^VIX"] = px["^VIX"].pct_change()
    rets["^TNX"] = px["^TNX"].diff()          # percentage points
    return px, rets, dates

def t0_index(dates, ev_date):
    d = pd.Timestamp(ev_date)
    pos = dates.searchsorted(d)
    if pos >= len(dates):
        return None
    return pos

def event_study(rets, dates, pos):
    n = len(dates)
    est_idx = np.arange(pos + EST_A, pos + EST_B + 1)
    win_idx = np.arange(pos + WIN_LO, pos + WIN_HI + 1)
    if est_idx[0] < 1 or win_idx[-1] >= n:   # need history for pct_change too
        return None
    spy = rets["SPY"].values
    out = {}
    for t in TICKERS:
        r = rets[t].values
        r_est, spy_est = r[est_idx], spy[est_idx]
        if t in BETA_ASSETS:
            beta = np.cov(r_est, spy_est)[0, 1] / np.var(spy_est)
            alpha = r_est.mean() - beta * spy_est.mean()
            resid = r_est - (alpha + beta * spy_est)
            resid_win = r[win_idx] - (alpha + beta * spy[win_idx])
        else:
            mu = r_est.mean()
            resid = r_est - mu
            resid_win = r[win_idx] - mu
        sig = resid.std(ddof=1)
        rel = win_idx - pos
        def car(a, b):
            sel = (rel >= a) & (rel <= b)
            c = resid_win[sel].sum()
            k = sel.sum()
            tstat = c / (sig * np.sqrt(k)) if sig > 0 else np.nan
            return c, k, tstat, sig
        c11, k11, t11, _ = car(-1, 1)
        c210, k210, t210, _ = car(2, 10)
        out[t] = dict(CAR_11=c11, k11=k11, t11=t11, CAR_210=c210, k210=k210, t210=t210, sig=sig)
        if t in BETA_ASSETS:
            out[t]["beta"] = beta
    return out

def main():
    px, rets, dates = load_returns()
    rets = rets.dropna()
    dates = rets.index
    rows = []
    per_event = {}
    skipped = []
    for ev_date, label, typ, notes in EVENTS:
        pos = t0_index(dates, ev_date)
        if pos is None:
            skipped.append((ev_date, label))
            continue
        used = dates[pos].date().isoformat()
        res = event_study(rets, dates, pos)
        if res is None:
            skipped.append((ev_date, label, "insufficient history"))
            continue
        per_event[(ev_date, label)] = res
        row = dict(date=ev_date, used_date=used, label=label, type=typ, notes=notes)
        for t in TICKERS:
            for k in ("CAR_11", "t11", "CAR_210", "t210", "sig"):
                row[f"{t}_{k}"] = res[t][k]
        rows.append(row)
    ev_df = pd.DataFrame(rows)
    ev_df.to_csv("results/trump_events_per_event.csv", index=False)

    # --- Type aggregation (FED excludes 2019-08-23 confound) ---
    agg_rows = []
    for typ in ["tariff", "sanction", "energy", "fed"]:
        sub = ev_df[ev_df.type == typ]
        if typ == "fed":
            sub = sub[sub.date != "2019-08-23"]
        n = len(sub)
        for t in TICKERS:
            c11 = sub[f"{t}_CAR_11"].values
            c210 = sub[f"{t}_CAR_210"].values
            t11s = sub[f"{t}_t11"].values
            t210s = sub[f"{t}_t210"].values
            m11, m210 = c11.mean(), c210.mean()
            se11 = c11.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            se210 = c210.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            agg_rows.append(dict(
                type=typ, asset=t, n=n,
                mean_CAR_11=m11, se11=se11, t_cs_11=m11 / se11 if se11 else np.nan,
                mean_CAR_210=m210, se210=se210, t_cs_210=m210 / se210 if se210 else np.nan,
                mean_abs_CAR_11=np.abs(c11).mean(), mean_abs_CAR_210=np.abs(c210).mean(),
                frac_sig11=np.mean(np.abs(t11s) > 2), frac_sig210=np.mean(np.abs(t210s) > 2),
            ))
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv("results/trump_events_agg_by_type.csv", index=False)
    print("events:", len(ev_df), "skipped:", skipped)

    # --- quick printouts ---
    pd.set_option("display.width", 200)
    print("\n== per-event SPY CAR[-1,+1] / CAR[+2,+10] (%), t-stats ==")
    q = ev_df[["date", "label", "type", "SPY_CAR_11", "SPY_t11", "SPY_CAR_210", "SPY_t210"]].copy()
    for c in ["SPY_CAR_11", "SPY_CAR_210"]:
        q[c] = (q[c] * 100).round(2)
    for c in ["SPY_t11", "SPY_t210"]:
        q[c] = q[c].round(2)
    print(q.to_string(index=False))
    print("\n== type agg: mean CAR %% (t_cs) for SPY, XLE, USO, UUP, VIX%%, TNXpp ==")
    for _, r in agg_df.iterrows():
        if r["asset"] in ["SPY", "XLE", "USO", "UUP", "^VIX"]:
            f = 100
            u = "%"
        else:
            f = 1
            u = "pp"
        print(f"{r['type']:9s} {r['asset']:5s} n={int(r['n']):2d} "
              f"CAR11={r['mean_CAR_11']*f:+.2f}{u} (t={r['t_cs_11']:+.2f}) | "
              f"CAR210={r['mean_CAR_210']*f:+.2f}{u} (t={r['t_cs_210']:+.2f}) | "
              f"mean|CAR11|={r['mean_abs_CAR_11']*f:.2f}{u} | "
              f"frac|t|>2: {r['frac_sig11']:.2f}/{r['frac_sig210']:.2f}")

if __name__ == "__main__":
    main()
