# Trading Bot

Paper-trading-first algorithmic trading bot for **crypto** and **stocks/options/derivatives**.
Backtest on real data, paper trade for free, graduate to real money only when the
numbers earn it.

## What's inside

| Layer | Files | Notes |
|---|---|---|
| Strategies | `bot/strategies/` | `momentum_breakout` (day trading), `swing_trend` (swing), `options_momentum` (long equity options) |
| Brokers | `bot/brokers/` | `alpaca` (paper: stocks, options, crypto), `deribit` (testnet: perps/futures/options), `simulated` (backtests) |
| Risk | `bot/risk.py` | risk-per-trade sizing, 20% max position, 3% daily-loss halt, stop-loss/take-profit on every trade |
| Backtest | `bot/backtest.py` | next-bar-open fills (no lookahead), slippage, Sharpe / max-DD / win-rate / profit-factor |

**Live trading is hard-gated.** The code refuses to touch a live venue unless
`LIVE_TRADING_ENABLED=true`. Paper is the default and only mode until you say otherwise.

## Quick start

```bash
cd trading-bot
pip install -r requirements.txt

# 1. Backtest on real BTC data (no keys needed)
python scripts/run_backtest.py --symbol BTCUSDT --strategy momentum --bars 3000
python scripts/run_backtest.py --symbol ETHUSDT --strategy swing --bars 3000
```

## ML pipeline (the core)

Built like a bioinformatics workflow — raw observations → features → labels → model → validation:

```bash
# train + walk-forward validate on BTC (no keys needed)
python scripts/train_model.py --symbol BTCUSDT --bars 5000 --horizon 12
```

1. **Features** (`bot/ml/features.py`): multi-horizon returns, RSI, EMA structure,
   ATR/volatility regime, volume z-scores, range position, time-of-day — all
   lookahead-free.
2. **Labels** (`bot/ml/labels.py`): triple-barrier labeling — +1 if the
   take-profit barrier is touched first, -1 if stop-loss, 0 if neither within
   the horizon. Barriers scale with ATR.
3. **Validation** (`bot/ml/model.py`): expanding-window walk-forward. The model
   only ever trades data it has never seen — the time-series analogue of CV.
   Reports out-of-sample accuracy, per-class precision, and a simulated
   strategy return/Sharpe.
4. **Execution**: the trained model plugs into the bot as a strategy:

```bash
python scripts/run_paper.py --broker alpaca --symbols BTC/USD --strategy ml \
    --model-path models/ml_BTCUSDT_1h.joblib --timeframe 1h --loop 3600
```

Retrain on a schedule (weekly) as regimes shift — models decay like everything
else in markets.

## Paper trading (free)

1. **Stocks/options/crypto:** create a free account at [alpaca.markets](https://alpaca.markets),
   generate **paper** API keys, and put them in `.env` (copy from `.env.example`).
2. **Crypto derivatives:** create a free account at [test.deribit.com](https://test.deribit.com),
   generate testnet API keys (`read` + `trade` scopes), add to `.env`.

```bash
# stocks swing trading on Alpaca paper, one pass (cron-friendly)
python scripts/run_paper.py --broker alpaca --symbols AAPL,SPY,QQQ --strategy swing --timeframe 1d

# equity options momentum on Alpaca paper (buys 7-30 DTE ~0.5-delta calls/puts)
python scripts/run_paper.py --broker alpaca --symbols SPY,QQQ --strategy options_momentum --timeframe 1h

# crypto perps momentum on Deribit testnet, looping every hour
python scripts/run_paper.py --broker deribit --symbols BTC-PERPETUAL --strategy momentum --timeframe 1h --loop 3600
```

Run it on a schedule (cron) rather than leaving a process up: one invocation =
one check-and-trade pass.

## Going live (later)

When paper results are consistently green: flip Alpaca to the live base URL or
Deribit to mainnet, fund a **small** account, and set `LIVE_TRADING_ENABLED=true`.
Keep the daily-loss halt and small position sizes — the bot enforces them either way.

## Disclaimer

This is educational software, not financial advice. Backtests are not predictive;
most day-trading strategies lose money after costs. Never trade money you can't
afford to lose.
