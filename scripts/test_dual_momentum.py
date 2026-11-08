"""
Strategy test T3 — Dual Momentum (Antonacci 2014), pre-registered single parameter set.

PRE-REGISTRATION (locked before running):
  Universe: SPY (US equity), EFA (ex-US equity), AGG (US aggregate bonds).
  Risk-free: ^IRX (13-week T-bill yield) from yfinance. FRED requires an API key,
             so the ^IRX proxy is used instead. Documented here, not changed post-hoc.
  Lookback: 12 months, single value, NO grid search (Sullivan/Timmermann/White caution).
  Rebalance: monthly, on month-end close. Position for month t+1 decided at close of t.
  Rule: pick max(12m total return SPY, 12m total return EFA); if winner's 12m return
        > 12m T-bill return, hold winner; else hold AGG.
  Costs: 5 bps per side; a switch of held asset = sell + buy = 10 bps.
  All prices: adjusted closes (total return, dividends reinvested).

Honesty pipeline: no lookahead (all signals use data known at month-end t),
regime splits including 2008 and 2020, benchmarks over identical windows,
CAPM beta/alpha + residual Sharpe (Sharpe net of equity beta).
"""
import numpy as np
import pandas as pd
import yfinance as yf

COST_PER_SIDE = 0.0005  # 5 bps
LOOKBACK = 12           # months, pre-registered, single value

# ---------------------------------------------------------------- data
px = yf.download(["SPY", "EFA", "AGG", "^IRX"], interval="1mo", period="max",
                 auto_adjust=True, progress=False)["Close"]
px = px.dropna(how="all")
irx = px.pop("^IRX") / 100.0          # annualized T-bill yield as decimal
r_bill_m = (1 + irx) ** (1 / 12) - 1  # monthly T-bill return

start = px.dropna().index[0]
print(f"Raw sample: {px.index[0].date()} .. {px.index[-1].date()}")
print(f"Common-data start (AGG inception): {start.date()}")
print(f"T-bill source: ^IRX (13w yield) via yfinance, FRED key not used (documented pre-reg choice)")

# ------------------------------------------------------- signal + backtest
r12 = px / px.shift(LOOKBACK) - 1                       # 12m total returns
bill12 = (1 + r_bill_m).rolling(LOOKBACK).apply(np.prod, raw=True) - 1  # 12m T-bill return

rel_winner = r12[["SPY", "EFA"]].idxmax(axis=1)          # relative momentum leg
win_r12 = r12[["SPY", "EFA"]].max(axis=1)
in_equity = win_r12 > bill12                            # absolute momentum leg
choice = pd.Series(np.where(in_equity, rel_winner, "AGG"), index=px.index)

# position for month t+1 decided at close of t
pos = choice.shift(1)
valid = pos.notna() & r12["SPY"].notna() & r_bill_m.notna()
pos = pos[valid]

# HONEST SAMPLE: the strategy cannot be executed before its safe leg exists.
# Start 12 months after the LAST-incepted asset (AGG, 2003-09) so every 12m
# return used by the rule is computable from real data.
common_start = px[["SPY", "EFA", "AGG"]].dropna().index[0]
test_start = (common_start + pd.DateOffset(months=12)).strftime("%Y-%m")
pos = pos.loc[test_start:]
assert not px.loc[pos.index, ["SPY", "EFA", "AGG"]].isna().any().any(), \
    "NaN prices inside test window — sample bug"
r_m = px.pct_change().loc[pos.index]   # monthly total returns, aligned to held months
assert not r_m.isna().any().any(), "NaN returns inside test window — sample bug"

port_r = r_m.lookup(pos.index, pos.values) if hasattr(r_m, "lookup") else \
    r_m.to_numpy()[np.arange(len(pos)), r_m.columns.get_indexer(pos.values)]
port_r = pd.Series(port_r, index=pos.index)

# costs: 10 bps whenever the held asset changes (sell + buy), incl. first entry
switched = pos.ne(pos.shift(1))
costs = switched.astype(float) * 2 * COST_PER_SIDE
port_r_net = port_r - costs

equity = (1 + port_r_net).cumprod()
spy_eq = (1 + r_m["SPY"]).cumprod()
# 60/40 SPY/AGG, monthly rebalanced, no costs (documented; conservative for the benchmark)
mix_r = 0.6 * r_m["SPY"] + 0.4 * r_m["AGG"]
mix_eq = (1 + mix_r).cumprod()

turnover_yr = switched.sum() / (len(pos) / 12)
in_agg = (pos == "AGG").mean()
print(f"Test window: {pos.index[0].date()} .. {pos.index[-1].date()} "
      f"({len(pos)} months, {switched.sum()} switches, {turnover_yr:.1f}/yr, "
      f"time in AGG: {in_agg:.1%})")

# --------------------------------------------------------------- metrics
def metrics(r_net, r_bill, label):
    ex = r_net - r_bill
    n = len(r_net)
    yrs = n / 12
    eq = (1 + r_net).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = ex.mean() / ex.std() * np.sqrt(12) if ex.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"label": label, "CAGR": cagr, "Sharpe": sharpe, "MaxDD": dd,
            "EndEq": eq.iloc[-1], "n_months": n}

def capm(r_net, r_bill, r_mkt):
    ex, mk = (r_net - r_bill).dropna(), (r_mkt - r_bill).dropna()
    idx = ex.index.intersection(mk.index)
    ex, mk = ex.loc[idx], mk.loc[idx]
    beta = ex.cov(mk) / mk.var()
    alpha_m = ex.mean() - beta * mk.mean()
    resid = ex - beta * mk
    return {"beta": beta, "alpha_ann": (1 + alpha_m) ** 12 - 1,
            "resid_sharpe": resid.mean() / resid.std() * np.sqrt(12),
            "corr": ex.corr(mk)}

bill = r_bill_m.loc[pos.index]
spy_r = r_m["SPY"]
res = []
for name, r in [("DualMom", port_r_net), ("SPY buy-hold", spy_r), ("60/40", mix_r)]:
    res.append(metrics(r, bill, name))
res = pd.DataFrame(res).set_index("label")
res["ExcessCAGR_vs_SPY"] = res["CAGR"] - res.loc["SPY buy-hold", "CAGR"]
res["ExcessCAGR_vs_6040"] = res["CAGR"] - res.loc["60/40", "CAGR"]
cap = capm(port_r_net, bill, spy_r)

print("\n=== FULL SAMPLE ===")
print(res.round(4).to_string())
print(f"\nCAPM vs SPY: beta={cap['beta']:.2f}  alpha(ann)={cap['alpha_ann']:.2%}  "
      f"residual Sharpe={cap['resid_sharpe']:.2f}  corr={cap['corr']:.2f}")

# regime splits (pre-registered windows)
regimes = {
    "GFC (2007-10..2009-12)": ("2007-10", "2009-12"),
    "COVID (2020-01..2021-12)": ("2020-01", "2021-12"),
    "First half (2004-09..2015-06)": ("2004-09", "2015-06"),
    "Second half (2015-07..2026-09)": ("2015-07", "2026-09"),
}
reg_rows = []
for rname, (a, b) in regimes.items():
    seg = slice(a, b)
    pr, sp, mx, bl = port_r_net.loc[seg], spy_r.loc[seg], mix_r.loc[seg], bill.loc[seg]
    if len(pr) < 6:
        continue
    eq_p, eq_s = (1 + pr).cumprod(), (1 + sp).cumprod()
    reg_rows.append({
        "regime": rname, "n": len(pr),
        "DualMom_tot": eq_p.iloc[-1] - 1, "SPY_tot": eq_s.iloc[-1] - 1,
        "excess_vs_SPY": eq_p.iloc[-1] - eq_s.iloc[-1],
        "DualMom_MaxDD": (eq_p / eq_p.cummax() - 1).min(),
        "SPY_MaxDD": (eq_s / eq_s.cummax() - 1).min(),
        "time_in_AGG": (pos.loc[seg] == "AGG").mean(),
    })
reg_df = pd.DataFrame(reg_rows).set_index("regime")
print("\n=== REGIMES ===")
print(reg_df.round(4).to_string())

# skeptic diagnostics: where did the return come from?
agg_r = port_r_net[pos == "AGG"]
eq_r = port_r_net[pos != "AGG"]
print(f"\nMonths in AGG: {(pos=='AGG').sum()} ({in_agg:.1%}); "
      f"avg monthly net when in AGG: {agg_r.mean():.3%}, when in equity: {eq_r.mean():.3%}")
print(f"Worst single month (strategy): {port_r_net.min():.2%} "
      f"({port_r_net.idxmin().date()}); SPY worst: {spy_r.min():.2%}")

# ledger
ledger = pd.DataFrame({"position": pos, "ret_gross": port_r, "cost": costs,
                       "ret_net": port_r_net, "SPY": spy_r, "mix6040": mix_r,
                       "bill_m": bill})
ledger.to_csv("results/dual_momentum_ledger.csv")

# persist summary for the report
summary = {"full": res.round(4).to_dict(), "capm": {k: round(v, 4) for k, v in cap.items()},
           "regimes": reg_df.round(4).to_dict(),
           "switches": int(switched.sum()), "turnover_per_yr": round(turnover_yr, 2),
           "time_in_agg": round(float(in_agg), 4),
           "test_start": str(pos.index[0].date()), "test_end": str(pos.index[-1].date())}
import json
json.dump(summary, open("results/dual_momentum_summary.json", "w"), indent=2)
print("\nWrote results/dual_momentum_ledger.csv and results/dual_momentum_summary.json")
