"""Train and evaluate a PPO trading agent with strict walk-forward discipline.

Pipeline:
  1. Fetch hourly bars (Coinbase fallback; Binance is geo-blocked).
  2. Split chronologically: TRAIN (60%) / VAL (20%) / TEST (20%).
     Normalization stats are fit on TRAIN only.
  3. Train PPO on TRAIN. Use VAL once for a single principled
     hyperparameter comparison (two configs max).
  4. Evaluate the chosen policy ONCE on TEST (deterministic policy,
     with costs) and compare against buy-and-hold.

Usage:
  python3 scripts/train_rl.py --symbol BTCUSDT --bars 15000 --timesteps 300000 --seed 42

Seeds are fixed everywhere (env, numpy, torch, SB3) for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.data.binance import fetch_klines
from bot.rl.env import TradingEnv, buy_and_hold, FEE_RATE

SEED = 42
LOOKBACK = 24


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # perf; seeds still fixed
    except ImportError:
        pass


def make_env(bars: pd.DataFrame, mean: np.ndarray, std: np.ndarray,
             seed: int) -> TradingEnv:
    env = TradingEnv(bars, lookback=LOOKBACK, fee_rate=FEE_RATE, seed=seed)
    env.set_normalization(mean, std)
    return env


def evaluate(model, bars: pd.DataFrame, mean: np.ndarray, std: np.ndarray,
             seed: int, label: str) -> dict:
    env = make_env(bars, mean, std, seed)
    stats = env.run_policy(model)
    bh = buy_and_hold(bars)
    print(f"[{label}] policy: ret={stats['total_return']:+.2%} "
          f"sharpe={stats['sharpe']:+.2f} dd={stats['max_drawdown']:.2%} "
          f"trades={stats['trades']} turnover={stats['turnover']:.0f}x", flush=True)
    print(f"[{label}] buy-hold: ret={bh['total_return']:+.2%} "
          f"sharpe={bh['sharpe']:+.2f} dd={bh['max_drawdown']:.2%}", flush=True)
    stats["buy_hold"] = bh
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--bars", type=int, default=15000)
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="models/rl_ppo_BTCUSDT_1h.zip")
    args = ap.parse_args()
    set_seeds(args.seed)

    print(f"Fetching {args.bars} hourly bars for {args.symbol}...", flush=True)
    bars = fetch_klines(args.symbol, "1h", total=args.bars)
    print(f"  got {len(bars)} bars: {bars.index[0]} -> {bars.index[-1]}", flush=True)

    n = len(bars)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    train_bars, val_bars, test_bars = bars.iloc[:i1], bars.iloc[i1:i2], bars.iloc[i2:]
    print(f"  split: train={len(train_bars)} val={len(val_bars)} test={len(test_bars)}",
          flush=True)

    # normalization fit on TRAIN only (no leakage into val/test)
    tmp = make_env(train_bars, np.zeros(1), np.ones(1), args.seed)
    mean = tmp.features.mean(axis=0)
    std = tmp.features.std(axis=0)
    print(f"  features: {tmp.features.shape[1]} (standardized on train)", flush=True)

    # --- train candidate configs; compare ONCE on VAL ---
    configs = {
        "A_default": {"net_arch": [64, 64], "learning_rate": 3e-4},
        "B_wide": {"net_arch": [128, 128], "learning_rate": 1e-4},
    }
    val_scores = {}
    models = {}
    for name, kw in configs.items():
        print(f"Training {name} ({args.timesteps} steps)...", flush=True)
        env = make_env(train_bars, mean, std, args.seed)
        lr = kw.pop("learning_rate")
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, seed=args.seed, verbose=0,
                    policy_kwargs={"net_arch": kw["net_arch"]},
                    n_steps=1024, batch_size=256, n_epochs=4,
                    learning_rate=lr, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2)
        t0 = time.time()
        model.learn(total_timesteps=args.timesteps)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
        models[name] = model
        v = evaluate(model, val_bars, mean, std, args.seed + 1, f"VAL/{name}")
        # selection metric: Sharpe net of costs (principled, fixed before seeing val)
        val_scores[name] = v["sharpe"]

    best = max(val_scores, key=val_scores.get)
    print(f"VAL selection: {best} (sharpe {val_scores[best]:+.2f})", flush=True)
    model = models[best]
    model.save(args.out)
    print(f"Saved policy -> {args.out}", flush=True)

    # --- TEST: touched exactly once ---
    print("=== FINAL TEST (held out) ===", flush=True)
    test_stats = evaluate(model, test_bars, mean, std, args.seed + 2, "TEST")

    # persist summary
    summary = {
        "symbol": args.symbol, "bars": args.bars, "seed": args.seed,
        "timesteps": args.timesteps, "lookback": LOOKBACK,
        "fee_rate": FEE_RATE, "reward": "pos_prev * ret_simple - cost_frac",
        "train_range": [str(train_bars.index[0]), str(train_bars.index[-1])],
        "val_range": [str(val_bars.index[0]), str(val_bars.index[-1])],
        "test_range": [str(test_bars.index[0]), str(test_bars.index[-1])],
        "val_sharpe": val_scores, "chosen": best,
        "test": {k: v for k, v in test_stats.items() if k != "equity_curve"},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/rl_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Wrote results/rl_summary.json", flush=True)


if __name__ == "__main__":
    main()
