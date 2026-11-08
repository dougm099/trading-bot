"""Gymnasium trading environment: hourly crypto bars, discrete actions.

Design notes (deliberate choices, documented so reward-shaping can't hide):
- Actions are *target positions* {flat, long, short}, not deltas. This keeps
  the policy interpretable: "what should my exposure be right now?"
- Position is a fraction of equity in {-1, 0, +1} (full-size long/short/flat).
- Reward per step = position_return - trade_cost, both as fractions of
  equity. No extra shaping bonuses, no profit-target hacks.
- Costs are charged on the *full notional traded* at 10 bps/side, which is
  itself the turnover penalty: flipping long->short costs 20 bps.
- Episodes are fixed historical windows; the position is force-closed at the
  end of the window so the agent can't hide losses in an open position.
- All features are lookahead-free (computed from bars at/before time t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from ..ml.features import build_features

ACTION_TO_POS = {0: 0.0, 1: 1.0, 2: -1.0}
FEE_RATE = 0.001  # 10 bps per side


class TradingEnv(gym.Env):
    """Trade one crypto pair on hourly bars.

    Observation: flattened lookback window of standardized features,
    plus [current_position, unrealized_pnl].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        lookback: int = 24,
        fee_rate: float = FEE_RATE,
        seed: int | None = None,
    ):
        super().__init__()
        self.lookback = lookback
        self.fee_rate = fee_rate

        closes = bars["close"].astype(float)
        feats = build_features(bars).dropna()
        # align closes to the feature index (features have NaN warmup rows)
        closes = closes.loc[feats.index]
        self.closes = closes.to_numpy()
        self.feat_names = list(feats.columns)
        self.features = feats.to_numpy(dtype=np.float64)

        # standardization stats are set externally (fit on TRAIN only)
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None

        n_feat = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lookback * n_feat + 2,), dtype=np.float64,
        )
        self.action_space = spaces.Discrete(3)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ utils
    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.feat_mean = mean
        self.feat_std = np.where(std < 1e-8, 1.0, std)

    def _obs(self) -> np.ndarray:
        w = self.features[self.t - self.lookback + 1: self.t + 1]
        w = (w - self.feat_mean) / self.feat_std
        w = np.clip(w, -10.0, 10.0)
        unrl = self.position * np.log(self.closes[self.t] / self.entry_price) \
            if self.position != 0 else 0.0
        return np.concatenate([w.ravel(), [self.position, unrl]]).astype(np.float64)

    # ------------------------------------------------------------------ gym api
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback - 1
        self.position = 0.0
        self.entry_price = 1.0
        self.equity = 1.0
        self.equity_curve = [1.0]
        self.trades = 0
        self.turnover = 0.0  # cumulative notional traded / starting equity
        return self._obs(), {}

    def step(self, action: int):
        target = ACTION_TO_POS[int(action)]
        prev_pos = self.position
        price_now = self.closes[self.t]
        price_next = self.closes[self.t + 1]

        # execute target position at current close; pay fees on notional traded
        traded = abs(target - prev_pos) * self.equity
        cost = traded * self.fee_rate
        if target != prev_pos:
            self.trades += 1
            self.turnover += abs(target - prev_pos)
            if target != 0.0:
                self.entry_price = price_now

        # hold prev position over [t, t+1]
        ret = price_next / price_now - 1.0
        pnl = prev_pos * ret * self.equity
        self.equity = self.equity + pnl - cost
        self.equity = max(self.equity, 1e-9)  # avoid negative equity in sim
        self.equity_curve.append(self.equity)

        # reward = step P&L as fraction of equity, net of costs
        reward = prev_pos * ret - cost / max(self.equity, 1e-9)
        self.position = target
        self.t += 1
        terminated = False
        truncated = self.t >= len(self.closes) - 1
        if truncated and self.position != 0.0:
            # force-flat at window end so no open-position losses can hide
            close_cost = abs(self.position) * self.equity * self.fee_rate
            self.equity -= close_cost
            self.equity_curve[-1] = self.equity
            self.trades += 1
            self.turnover += abs(self.position)
            self.position = 0.0
        return self._obs(), float(reward), terminated, truncated, {}

    # ------------------------------------------------------------------ eval
    def run_policy(self, policy) -> dict:
        """Roll out a deterministic SB3-style policy; return stats dict."""
        obs, _ = self.reset()
        done = False
        rets = []
        prev_eq = 1.0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, _, truncated, _ = self.step(int(action))
            done = truncated
            rets.append(self.equity / prev_eq - 1)
            prev_eq = self.equity
        rets = np.array(rets)
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        # hourly bars -> annualize with 24*365
        sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(24 * 365) \
            if len(rets) > 1 else 0.0
        return {
            "total_return": eq[-1] - 1.0,
            "sharpe": float(sharpe),
            "max_drawdown": float(dd.min()),
            "trades": self.trades,
            "turnover": float(self.turnover),
            "n_bars": len(rets),
            "equity_curve": eq,
        }


def buy_and_hold(bars: pd.DataFrame, fee_rate: float = FEE_RATE) -> dict:
    """Long-only buy-and-hold baseline over the same window, with costs."""
    c = bars["close"].astype(float).to_numpy()
    gross = c[-1] / c[0] - 1.0
    net = (1 + gross) * (1 - fee_rate) - 1.0  # in + out
    rets = c[1:] / c[:-1] - 1.0
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(24 * 365)
    eq = c / c[0]
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    return {"total_return": float(net), "sharpe": float(sharpe),
            "max_drawdown": float(dd), "trades": 2, "turnover": 2.0}
