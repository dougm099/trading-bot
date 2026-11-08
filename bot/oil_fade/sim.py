"""Event-driven simulator for the oil-fade strategy.

Replays 5-minute bars through OilFadeStrategy with an explicit cost model:
  - entry_slip_bps: adverse slippage on the trigger-bar close fill
  - stop_slip_bps:  adverse slippage beyond the stop level on stop-outs
  - exit_slip_bps:  adverse slippage on the time-exit fill
  - cost_bps:       round-trip commission/fees, applied per round trip
  - borrow_bps_day: stock-loan cost prorated by holding time (fraction of day)

All slippage is adverse by construction (conservative). P&L is reported in
basis points of entry notional so configs are comparable regardless of size;
run_equity() converts to a dollar equity curve under fixed-fractional risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .strategy import Bar, OilFadeConfig, OilFadeStrategy


@dataclass
class SimCosts:
    entry_slip_bps: float = 2.0
    stop_slip_bps: float = 3.0
    exit_slip_bps: float = 1.0
    cost_bps: float = 4.0
    borrow_bps_day: float = 0.5


def _to_bars(df: pd.DataFrame) -> List[Bar]:
    """df: datetime index (ET), columns o/h/l/c."""
    return [Bar(ts.to_pydatetime(), float(r.o), float(r.h), float(r.l), float(r.c))
            for ts, r in df.iterrows()]


def run_day(df: pd.DataFrame, cfg: OilFadeConfig,
            costs: SimCosts = SimCosts()) -> Dict:
    """Simulate one session. Returns a trade dict (empty if no trigger)."""
    bars = _to_bars(df)
    if not bars:
        return {}
    strat = OilFadeStrategy(cfg)
    # gap needs prior close: caller passes it via df.attrs
    strat.new_day(session_open=bars[0].open, prior_close=df.attrs.get("prior_close"))

    trade: Dict = {"triggered": False}
    for b in bars:
        sig = strat.on_bar(b)
        if sig.action == "short":
            fill = b.close * (1 - costs.entry_slip_bps / 1e4)
            trade.update({"triggered": True, "entry": fill, "entry_ts": b.ts,
                          "stop_level": strat.stop})
        elif sig.action == "cover" and trade.get("triggered") and "exit" not in trade:
            if sig.reason == "stop":
                fill = sig.price * (1 + costs.stop_slip_bps / 1e4)
            else:
                fill = b.close * (1 + costs.exit_slip_bps / 1e4)
            hold_frac = (b.ts - trade["entry_ts"]).total_seconds() / (6.5 * 3600)
            gross = (trade["entry"] - fill) / trade["entry"] * 1e4
            net = gross - costs.cost_bps - costs.borrow_bps_day * hold_frac
            trade.update({"exit": fill, "exit_ts": b.ts, "exit_why": sig.reason,
                          "pnl_bps": round(net, 1)})
    if trade.get("triggered") and "exit" not in trade:
        # never exited (e.g. data ends early): force-cover at last close
        b = bars[-1]
        fill = b.close * (1 + costs.exit_slip_bps / 1e4)
        hold_frac = (b.ts - trade["entry_ts"]).total_seconds() / (6.5 * 3600)
        gross = (trade["entry"] - fill) / trade["entry"] * 1e4
        trade.update({"exit": fill, "exit_ts": b.ts, "exit_why": "forced",
                      "pnl_bps": round(gross - costs.cost_bps, 1)})
    return trade


def run_backtest(days: List[pd.DataFrame], cfg: OilFadeConfig,
                 costs: SimCosts = SimCosts()) -> pd.DataFrame:
    rows = []
    for df in days:
        t = run_day(df, cfg, costs)
        rows.append({
            "date": str(df.attrs.get("date", "")),
            "triggered": t.get("triggered", False),
            "pnl_bps": t.get("pnl_bps", np.nan),
            "exit_why": t.get("exit_why", ""),
            "gap_pct": df.attrs.get("gap_pct", np.nan),
        })
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> Dict:
    t = trades[trades["triggered"]].copy()
    n = len(t)
    base = {"n": n, "win_rate": 0.0, "mean_bps": 0.0, "total_bps": 0.0,
            "sharpe_per_trade": 0.0, "t_stat": 0.0, "best": 0.0, "worst": 0.0,
            "profit_factor": 0.0, "stops": 0}
    if n == 0:
        return base
    p = t["pnl_bps"].to_numpy(float)
    wins = p[p > 0]
    losses = p[p <= 0]
    mean, sd = p.mean(), p.std(ddof=1) if n > 1 else 0.0
    return {
        "n": n,
        "win_rate": round(float((p > 0).mean()), 3),
        "mean_bps": round(float(mean), 1),
        "total_bps": round(float(p.sum()), 1),
        "sharpe_per_trade": round(float(mean / sd), 2) if sd > 0 else 0.0,
        "t_stat": round(float(mean / (sd / np.sqrt(n))), 2) if sd > 0 else 0.0,
        "best": round(float(p.max()), 1),
        "worst": round(float(p.min()), 1),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() != 0 else float("inf"),
        "stops": int((t["exit_why"] == "stop").sum()),
    }


def run_equity(days: List[pd.DataFrame], cfg: OilFadeConfig,
               costs: SimCosts = SimCosts(),
               start_equity: float = 100_000.0,
               max_position_pct: float = 150.0) -> pd.DataFrame:
    """Dollar equity curve: risk risk_per_trade_pct of equity per trade,
    sized so a stop-out loses ~risk_pct (stop distance + costs), subject to
    the same max-position cap the live runner enforces. NOTE: 1% risk with a
    0.75% stop implies ~133% notional short exposure."""
    eq = start_equity
    curve = []
    for df in days:
        t = run_day(df, cfg, costs)
        if t.get("triggered"):
            risk_frac = cfg.risk_per_trade_pct / 100
            stop_dist_frac = cfg.stop_pct / 100 + (costs.cost_bps + costs.stop_slip_bps) / 1e4
            # notional such that a stop-out loses ~risk_frac of equity
            notional = risk_frac * eq / stop_dist_frac if stop_dist_frac > 0 else 0
            notional = min(notional, max_position_pct / 100 * eq)
            eq += t["pnl_bps"] / 1e4 * notional
        curve.append({"date": str(df.attrs.get("date", "")), "equity": round(eq, 2)})
    out = pd.DataFrame(curve)
    out["drawdown_pct"] = (out["equity"] / out["equity"].cummax() - 1) * 100
    return out
