"""PaperAccount: a tiny simulated brokerage account.

Cash + share positions, explicit order fills at given prices, per-side
commission in bps on traded notional, daily mark-to-market, full trade log.
No broker, no network: pure arithmetic. Reusable by any strategy sim.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DUST_NOTIONAL = 1.0  # ignore orders smaller than $1


class PaperAccount:
    def __init__(self, cash: float = 100_000.0, cost_bps: float = 10.0):
        self.starting_cash = float(cash)
        self.cash = float(cash)
        self.cost_rate = cost_bps / 1e4
        self.positions: Dict[str, float] = {}   # ticker -> shares
        self._last_px: Dict[str, float] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []
        self.warnings: List[str] = []

    # ---------- pricing ----------
    def _px(self, ticker: str, prices: pd.Series) -> Optional[float]:
        p = prices.get(ticker, np.nan)
        if p is None or (isinstance(p, float) and (np.isnan(p) or p <= 0)):
            p = self._last_px.get(ticker)
            if p is None:
                self.warnings.append(f"no price for {ticker}")
                return None
            self.warnings.append(f"stale price for {ticker}")
        else:
            p = float(p)
            self._last_px[ticker] = p
        return p

    def equity(self, prices: pd.Series) -> float:
        val = self.cash
        for t, sh in self.positions.items():
            p = self._px(t, prices)
            if p:
                val += sh * p
        return val

    # ---------- orders ----------
    def _record(self, date, ticker, side, shares, price, cost):
        self.trades.append({"date": date, "ticker": ticker, "side": side,
                            "shares": round(shares, 6), "price": round(price, 4),
                            "notional": round(shares * price, 2),
                            "cost": round(cost, 2),
                            "cash_after": round(self.cash, 2)})

    def order_to_value(self, ticker: str, target_value: float,
                       price: float, date) -> float:
        """Move position toward target dollar value. Returns traded notional."""
        cur_val = self.positions.get(ticker, 0.0) * price
        delta = target_value - cur_val
        if abs(delta) < DUST_NOTIONAL:
            return 0.0
        if delta < 0:  # sell
            sell_val = min(-delta, cur_val)
            shares = sell_val / price
            cost = sell_val * self.cost_rate
            self.positions[ticker] = self.positions.get(ticker, 0.0) - shares
            if self.positions[ticker] < 1e-9:
                del self.positions[ticker]
            self.cash += sell_val - cost
            self._record(date, ticker, "SELL", shares, price, cost)
            return sell_val
        # buy, capped by available cash
        afford = self.cash / (1 + self.cost_rate)
        buy_val = min(delta, afford)
        if buy_val < DUST_NOTIONAL:
            return 0.0
        shares = buy_val / price
        cost = buy_val * self.cost_rate
        self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
        self.cash -= buy_val + cost
        self._record(date, ticker, "BUY", shares, price, cost)
        return buy_val

    def rebalance(self, targets: Dict[str, float], prices: pd.Series, date) -> float:
        """targets: ticker -> fraction of equity. Sells first, then buys
        (pro-rata scaled if cash binds). Returns total traded notional."""
        eq = self.equity(prices)
        desired = {t: w * eq for t, w in targets.items() if w > 0}
        traded = 0.0
        # sells first to free cash
        for t in list(self.positions):
            p = self._px(t, prices)
            if p is None:
                continue
            tgt = desired.get(t, 0.0)
            if tgt < self.positions[t] * p - DUST_NOTIONAL:
                traded += self.order_to_value(t, tgt, p, date)
        # buys, scaled if cash is short
        px_cache = {t: self._px(t, prices) for t in desired}
        px_cache = {t: p for t, p in px_cache.items() if p}
        need = sum(max(0.0, desired[t] - self.positions.get(t, 0.0) * px_cache[t])
                   for t in px_cache)
        scale = min(1.0, (self.cash / (1 + self.cost_rate)) / need) if need > 0 else 0.0
        for t, p in px_cache.items():
            cur = self.positions.get(t, 0.0) * p
            if desired.get(t, 0.0) > cur + DUST_NOTIONAL and scale > 0:
                tgt = cur + (desired[t] - cur) * scale
                traded += self.order_to_value(t, tgt, p, date)
        return traded

    def liquidate(self, prices: pd.Series, date) -> float:
        traded = 0.0
        for t in list(self.positions):
            p = self._px(t, prices)
            if p:
                traded += self.order_to_value(t, 0.0, p, date)
        return traded

    def mark(self, prices: pd.Series, date):
        row = {"date": date, "equity": self.equity(prices),
               "cash": self.cash, "n_positions": len(self.positions)}
        # same-date mark overwrites (e.g. pre-trade -> post-trade on rebalance day)
        if self.equity_curve and self.equity_curve[-1]["date"] == date:
            self.equity_curve[-1] = row
        else:
            self.equity_curve.append(row)

    # ---------- reporting ----------
    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve).set_index("date")

    def summary(self) -> dict:
        eq = self.equity_df()["equity"]
        r = eq.pct_change().dropna()
        nyr = (eq.index[-1] - eq.index[0]).days / 365.25
        dd = float((eq / eq.cummax() - 1).min())
        tr = self.trades_df()
        return {
            "start": round(self.starting_cash, 2),
            "end": round(float(eq.iloc[-1]), 2),
            "total_return": round(float(eq.iloc[-1] / eq.iloc[0] - 1), 4),
            "cagr": round(float((eq.iloc[-1] / eq.iloc[0]) ** (1 / nyr) - 1), 4) if nyr > 0 else 0.0,
            "sharpe_daily_ann": round(float(r.mean() / r.std() * math.sqrt(252)), 3) if r.std() > 0 else 0.0,
            "max_drawdown": round(dd, 4),
            "n_trades": len(tr),
            "total_cost": round(float(tr["cost"].sum()), 2) if len(tr) else 0.0,
            "n_warnings": len(self.warnings),
        }
