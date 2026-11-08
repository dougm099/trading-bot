"""Almgren-Chriss execution scheduling. This is cost reduction, not alpha.

Implements the AC (2001) sinh-optimal liquidation trajectory:

    x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)

where x(t) is the remaining position at time t, T the horizon, and the urgency

    kappa = sqrt(kyle_lambda * sigma^2 / eta)

with Kyle's lambda (permanent impact, estimated per bar as |return|/dollar
volume, after Kyle 1985 / Amihud 2002), sigma the return-volatility rate, and
eta the temporary-impact coefficient calibrated from the Bouchaud square-root
law so that a TWAP schedule reproduces  I(Q) ~= Y * sigma * sqrt(Q / Vbar).

TWAP is the kappa -> 0 special case and is available as an explicit fallback.

This module is PURE cost engineering: it changes *how* a parent order is
worked, never *whether* or *what* to trade. No signal content.

Honesty notes (see results/execution_ac.md):
- Paper/simulated fills do not model market impact or queue dynamics, so a
  simulated A/B of AC vs TWAP is *model-based* (square-root law + realized
  price paths). The honest measurement is forward paper: log implementation
  shortfall vs the arrival price and compare modes on real fills.
- The kappa used here follows the task/literature-note convention with Kyle's
  lambda as the urgency driver (illiquid -> urgent). Textbook AC uses a risk-
  aversion parameter there instead; the trajectory math is identical.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from .models import Fill, Order

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EXEC_STATE_PATH = RESULTS_DIR / "execution_state.json"
EXEC_FILLS_PATH = RESULTS_DIR / "execution_fills.csv"

# Clamp on kappa*T so the trajectory cannot collapse into a single slice.
_MAX_KAPPA_T = 6.0


@dataclass
class MarketImpact:
    """Calibrated per-second market-impact parameters for one symbol."""
    symbol: str
    sigma_per_s: float      # return volatility per second
    dollar_vol_per_s: float  # average notional volume per second ($/s)
    kyle_lambda: float      # permanent impact: expected |return| per $ traded
    bar_seconds: float
    lookback_bars: int


@dataclass
class ChildSlice:
    t_offset_s: float  # seconds after schedule start when this slice is due
    qty: float         # signed quantity (same sign as parent)
    frac: float        # fraction of parent quantity


@dataclass
class ExecutionSchedule:
    symbol: str
    side: str            # 'buy' | 'sell'
    total_qty: float     # signed
    arrival_price: float
    mode: str            # 'ac' | 'twap'
    horizon_s: float
    n_slices: int
    kappa: float
    eta: float
    kyle_lambda: float
    sigma_per_s: float
    slices: List[ChildSlice] = field(default_factory=list)
    # expected-cost analytics (model-based)
    exp_temp_cost_usd: float = 0.0
    exp_perm_cost_usd: float = 0.0   # schedule-independent; cancels in A/B
    timing_risk_usd: float = 0.0     # std of timing cost

    @property
    def notional(self) -> float:
        return abs(self.total_qty) * self.arrival_price

    def exp_shortfall_bps(self) -> float:
        v = self.notional
        return 1e4 * (self.exp_temp_cost_usd + self.exp_perm_cost_usd) / v if v > 0 else 0.0

    def timing_risk_bps(self) -> float:
        v = self.notional
        return 1e4 * self.timing_risk_usd / v if v > 0 else 0.0


def calibrate_market(bars: pd.DataFrame, symbol: str = "",
                     lookback: int = 60) -> MarketImpact:
    """Estimate per-second sigma, dollar volume rate, and Kyle's lambda.

    Kyle's lambda ~= mean(|return| / dollar_volume) over the lookback
    (Kyle 1985; Amihud 2002 illiquidity ratio). Bars: ascending datetime index,
    columns open/high/low/close/volume. Volume is in *base* units.
    """
    df = bars.tail(max(lookback + 1, 10)).copy()
    idx = pd.DatetimeIndex(df.index)
    bar_seconds = float(np.median(np.diff(idx.astype("int64") / 1e9)))
    bar_seconds = max(bar_seconds, 1.0)

    close = df["close"].astype(float)
    ret = close.pct_change().dropna().tail(lookback)
    dollar_vol = (close * df["volume"].astype(float)).reindex(ret.index)

    sigma_per_s = float(ret.std() / math.sqrt(bar_seconds)) if len(ret) > 1 else 0.0
    dollar_vol_per_s = float(dollar_vol.mean() / bar_seconds) if len(dollar_vol) else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (ret.abs() / dollar_vol).replace([np.inf, -np.inf], np.nan).dropna()
    kyle_lambda = float(lam.mean()) if len(lam) else 0.0

    return MarketImpact(
        symbol=symbol,
        sigma_per_s=max(sigma_per_s, 0.0),
        dollar_vol_per_s=max(dollar_vol_per_s, 1e-9),
        kyle_lambda=max(kyle_lambda, 0.0),
        bar_seconds=bar_seconds,
        lookback_bars=lookback,
    )


def _sinh_fracs(kappa: float, horizon_s: float, n_slices: int) -> List[float]:
    """Per-slice fractions of the parent from the AC sinh trajectory.

    Falls back to equal (TWAP) fractions as kappa -> 0.
    """
    if n_slices <= 1:
        return [1.0]
    if kappa <= 0 or not math.isfinite(kappa):
        return [1.0 / n_slices] * n_slices
    kt = min(kappa * horizon_s, _MAX_KAPPA_T)
    if kt < 1e-6:
        return [1.0 / n_slices] * n_slices
    denom = math.sinh(kt)
    ts = np.linspace(0.0, horizon_s, n_slices + 1)
    # remaining fraction at each grid point: sinh(k*(T-t))/sinh(k*T)
    rem = np.sinh(kt * (1.0 - ts / horizon_s)) / denom
    fracs = (rem[:-1] - rem[1:]).tolist()
    tot = sum(fracs)
    return [f / tot for f in fracs] if tot > 0 else [1.0 / n_slices] * n_slices


def build_schedule(symbol: str, side: str, qty: float, arrival_price: float,
                   bars: pd.DataFrame, mode: str = "ac",
                   horizon_s: float = 600.0, n_slices: int = 10,
                   lookback: int = 60, square_root_y: float = 1.0,
                   round_lot: Optional[float] = None) -> ExecutionSchedule:
    """Build a child-order schedule for a parent order.

    side: 'buy' | 'sell'; qty: unsigned quantity; arrival_price: decision price.
    mode: 'ac' (sinh-optimal) | 'twap' (equal slices, the kappa->0 limit).
    Calibration: eta from the square-root law so TWAP reproduces
        temp_impact ~= Y * sigma * sqrt(Q / Vbar).
    """
    assert side in ("buy", "sell")
    assert mode in ("ac", "twap")
    assert qty > 0 and arrival_price > 0 and horizon_s > 0 and n_slices >= 1

    mi = calibrate_market(bars, symbol=symbol, lookback=lookback)
    notional = qty * arrival_price
    sigma, vol_s, lam = mi.sigma_per_s, mi.dollar_vol_per_s, mi.kyle_lambda

    # Temporary-impact coefficient: equate TWAP linear impact eta*(V/T) to the
    # square-root law Y*sigma*sqrt(V/Vbar)  =>  eta = Y*sigma*T/sqrt(V*vol_s).
    eta = square_root_y * sigma * horizon_s / math.sqrt(max(notional, 1e-9) * vol_s)

    if mode == "twap" or lam <= 0 or sigma <= 0 or eta <= 0:
        kappa = 0.0
    else:
        kappa = math.sqrt(lam * sigma ** 2 / eta)
        kappa = min(kappa, _MAX_KAPPA_T / horizon_s)

    fracs = _sinh_fracs(kappa, horizon_s, n_slices)
    signed_qty = qty if side == "buy" else -qty
    slices: List[ChildSlice] = []
    for j, f in enumerate(fracs):
        q = signed_qty * f
        if round_lot:
            q = math.floor(abs(q) / round_lot) * round_lot * (1 if q >= 0 else -1)
        slices.append(ChildSlice(t_offset_s=horizon_s * (j + 1) / n_slices,
                                 qty=q, frac=f))
    # sweep rounding dust into the last slice so slices sum to the parent
    if round_lot:
        dust = signed_qty - sum(s.qty for s in slices)
        slices[-1].qty += dust

    sched = ExecutionSchedule(
        symbol=symbol, side=side, total_qty=signed_qty, arrival_price=arrival_price,
        mode=mode if kappa > 0 else "twap", horizon_s=horizon_s, n_slices=n_slices,
        kappa=kappa, eta=eta, kyle_lambda=lam, sigma_per_s=sigma, slices=slices,
    )

    # ---- expected-cost analytics (model-based) ----
    tau = horizon_s / n_slices
    temp = 0.0
    tvar = 0.0
    for j, s in enumerate(slices):
        qn = abs(s.qty) * arrival_price          # slice notional ($)
        rate = qn / tau                           # $/s
        temp += eta * rate * qn                    # eta*rate is relative concession
        # average remaining notional during slice j
        rem_before = sum(abs(x.qty) for x in slices[j:]) * arrival_price
        rem_after = rem_before - qn
        rem_avg = 0.5 * (rem_before + rem_after)
        tvar += (sigma ** 2) * (rem_avg ** 2) * tau
    sched.exp_temp_cost_usd = temp
    sched.exp_perm_cost_usd = 0.5 * lam * notional ** 2  # schedule-independent
    sched.timing_risk_usd = math.sqrt(max(tvar, 0.0))
    return sched


def implementation_shortfall_bps(side: str, fill_price: float,
                                 arrival_price: float) -> float:
    """Per-fill implementation shortfall in bps vs the arrival (decision) price."""
    if arrival_price <= 0:
        return 0.0
    return (1e4 * (fill_price - arrival_price) / arrival_price
            if side == "buy" else 1e4 * (arrival_price - fill_price) / arrival_price)


def simulate_realized_shortfall(schedule: ExecutionSchedule, bars: pd.DataFrame,
                                start_ts: pd.Timestamp,
                                spread_bps: float = 2.0) -> Dict[str, float]:
    """Offline A/B engine: work a schedule against *realized* future bars.

    Each slice executes at the first bar at/after its due time, at that bar's
    close plus half-spread plus modeled temporary impact (eta * rate). The
    timing component (whether the price drifted for/against us while working)
    comes from real data; only the impact concession is modeled. Returns
    realized shortfall in bps and its decomposition.
    """
    df = bars[bars.index >= start_ts]
    if len(df) == 0:
        return {"is_bps": 0.0, "timing_bps": 0.0, "impact_bps": 0.0, "slices_done": 0}
    side_sign = 1.0 if schedule.side == "buy" else -1.0
    p0 = schedule.arrival_price
    tau = schedule.horizon_s / schedule.n_slices
    is_usd = 0.0
    timing_usd = 0.0
    impact_usd = 0.0
    done = 0
    for s in schedule.slices:
        due = start_ts + pd.Timedelta(seconds=s.t_offset_s)
        fut = df[df.index >= due]
        if len(fut) == 0:
            break
        mid = float(fut["close"].iloc[0])
        qn = abs(s.qty) * p0
        rate = qn / tau
        concession_rel = spread_bps / 2e4 + schedule.eta * rate
        exec_px = mid * (1.0 + side_sign * concession_rel)
        is_usd += side_sign * (exec_px - p0) * abs(s.qty)
        timing_usd += side_sign * (mid - p0) * abs(s.qty)
        impact_usd += concession_rel * mid * abs(s.qty) * side_sign * side_sign
        done += 1
    v = schedule.notional
    return {
        "is_bps": 1e4 * is_usd / v if v else 0.0,
        "timing_bps": 1e4 * timing_usd / v if v else 0.0,
        "impact_bps": 1e4 * impact_usd / v if v else 0.0,
        "slices_done": done,
    }


# --------------------------------------------------------------------------
# Paper-path executor: persists schedules across cron invocations, routes
# child orders through the broker, and logs implementation shortfall per fill.
# --------------------------------------------------------------------------

class ACExecutor:
    """Opt-in execution agent for the paper-trading path.

    Usage:
        ex = ACExecutor(mode="ac", horizon_s=600, n_slices=10)
        ex.submit(order, bars)          # schedules child slices, persists state
        ex.work(broker)                # call each loop/cron tick: fires due slices
    Fills are appended to results/execution_fills.csv with IS vs arrival price,
    which is the honest forward A/B dataset (TWAP vs AC).
    """

    def __init__(self, mode: str = "ac", horizon_s: float = 600.0,
                 n_slices: int = 10, lookback: int = 60,
                 state_path: Path = EXEC_STATE_PATH,
                 fills_path: Path = EXEC_FILLS_PATH,
                 round_lot: Optional[float] = None):
        assert mode in ("ac", "twap")
        self.mode = mode
        self.horizon_s = horizon_s
        self.n_slices = n_slices
        self.lookback = lookback
        self.state_path = Path(state_path)
        self.fills_path = Path(fills_path)
        self.round_lot = round_lot
        self._active: List[Dict] = self._load()

    # ---- persistence ----
    def _load(self) -> List[Dict]:
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            return []

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._active, indent=1))

    # ---- scheduling ----
    def submit(self, order: Order, bars: pd.DataFrame,
               arrival_price: Optional[float] = None) -> str:
        """Schedule a parent order. Returns exec_id. Persists immediately."""
        px = arrival_price if arrival_price else (order.limit_price or 0.0)
        if px <= 0:
            raise ValueError("submit() needs arrival_price (or order.limit_price)")
        sched = build_schedule(order.symbol, order.side, abs(order.qty), px, bars,
                               mode=self.mode, horizon_s=self.horizon_s,
                               n_slices=self.n_slices, lookback=self.lookback,
                               round_lot=self.round_lot)
        exec_id = uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._active.append({
            "exec_id": exec_id, "symbol": order.symbol, "side": order.side,
            "asset_class": order.asset_class, "reduce_only": order.reduce_only,
            "arrival_price": px, "mode": sched.mode, "horizon_s": self.horizon_s,
            "submitted_ts": now, "kappa": sched.kappa, "eta": sched.eta,
            "slices": [{"t_offset_s": s.t_offset_s, "qty": s.qty, "frac": s.frac,
                        "done": False, "fill_price": None, "fill_ts": None}
                       for s in sched.slices],
        })
        self._save()
        return exec_id

    def pending(self) -> int:
        return sum(1 for e in self._active
                   for s in e["slices"] if not s["done"])

    def work(self, broker) -> List[Fill]:
        """Fire every due slice through broker.place_order(). Returns fills.

        A slice whose child order raises is left pending and retried on the
        next tick; the failure is recorded on the slice (visible in state)
        and logged (warning on first occurrence, debug on repeats) so a
        persistently failing execution (e.g. insufficient cash) is visible
        instead of silently stuck.
        """
        log = logging.getLogger("execution")
        fills: List[Fill] = []
        now = datetime.now(timezone.utc)
        dirty = False
        for e in self._active:
            t0 = datetime.fromisoformat(e["submitted_ts"])
            elapsed = (now - t0).total_seconds()
            for i, s in enumerate(e["slices"]):
                if s["done"] or s["t_offset_s"] > elapsed or abs(s["qty"]) < 1e-12:
                    continue
                child = Order(symbol=e["symbol"],
                              side=e["side"],
                              qty=abs(s["qty"]),
                              order_type="market",
                              asset_class=e.get("asset_class", "spot"),
                              reduce_only=bool(e.get("reduce_only", False)),
                              client_id=f"ac-{e['exec_id']}-{i}")
                try:
                    f: Fill = broker.place_order(child)
                except Exception as exc:  # retry on next work() tick
                    s["err_count"] = s.get("err_count", 0) + 1
                    s["last_error"] = f"{type(exc).__name__}: {exc}"
                    dirty = True
                    if s["err_count"] == 1:
                        log.warning("exec %s slice %d failed: %s",
                                    e["exec_id"], i, s["last_error"])
                    else:
                        log.debug("exec %s slice %d failed again (%d): %s",
                                  e["exec_id"], i, s["err_count"], s["last_error"])
                    continue
                s["done"] = True
                s["fill_price"] = f.price
                s["fill_ts"] = now.isoformat()
                dirty = True
                fills.append(f)
                self._log_fill(e, i, s, f)
        # drop fully-worked executions
        self._active = [e for e in self._active
                        if any(not s["done"] for s in e["slices"])]
        if dirty:
            self._save()
        return fills

    def _log_fill(self, e: Dict, i: int, s: Dict, f: Fill) -> None:
        self.fills_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.fills_path.exists() or self.fills_path.stat().st_size == 0
        is_bps = implementation_shortfall_bps(e["side"], f.price, e["arrival_price"])
        with self.fills_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["ts_utc", "exec_id", "symbol", "side", "mode",
                            "slice_idx", "qty", "arrival_price", "fill_price",
                            "is_bps", "horizon_s", "kappa"])
            w.writerow([datetime.now(timezone.utc).isoformat(), e["exec_id"],
                        e["symbol"], e["side"], e["mode"], i, abs(s["qty"]),
                        e["arrival_price"], f.price, round(is_bps, 3),
                        e["horizon_s"], round(e["kappa"], 6)])

    def cancel(self, exec_id: str) -> bool:
        before = len(self._active)
        self._active = [e for e in self._active if e["exec_id"] != exec_id]
        self._save()
        return len(self._active) < before
