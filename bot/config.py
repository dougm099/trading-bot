"""Configuration loaded from environment variables.

Copy .env.example to .env and fill in keys. Paper trading is the default and
only enabled mode; live trading is hard-gated (see LIVE_TRADING_ENABLED).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    # ---- mode ----
    mode: str = field(default_factory=lambda: _get("TRADING_MODE", "backtest"))  # backtest | paper
    live_trading_enabled: bool = field(default_factory=lambda: _get("LIVE_TRADING_ENABLED", "false").lower() == "true")

    # ---- alpaca (stocks + options + crypto), paper by default ----
    alpaca_key: str = field(default_factory=lambda: _get("ALPACA_API_KEY"))
    alpaca_secret: str = field(default_factory=lambda: _get("ALPACA_SECRET_KEY"))
    alpaca_base_url: str = field(default_factory=lambda: _get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))

    # ---- deribit testnet (crypto derivatives) ----
    deribit_client_id: str = field(default_factory=lambda: _get("DERIBIT_CLIENT_ID"))
    deribit_client_secret: str = field(default_factory=lambda: _get("DERIBIT_CLIENT_SECRET"))
    deribit_testnet: bool = field(default_factory=lambda: _get("DERIBIT_TESTNET", "true").lower() == "true")

    # ---- risk ----
    starting_equity: float = field(default_factory=lambda: _get_float("STARTING_EQUITY", 10_000))
    risk_per_trade_pct: float = field(default_factory=lambda: _get_float("RISK_PER_TRADE_PCT", 1.0))
    max_position_pct: float = field(default_factory=lambda: _get_float("MAX_POSITION_PCT", 20.0))
    max_daily_loss_pct: float = field(default_factory=lambda: _get_float("MAX_DAILY_LOSS_PCT", 3.0))
    max_open_positions: int = field(default_factory=lambda: _get_int("MAX_OPEN_POSITIONS", 5))
    stop_loss_pct: float = field(default_factory=lambda: _get_float("STOP_LOSS_PCT", 2.0))
    take_profit_pct: float = field(default_factory=lambda: _get_float("TAKE_PROFIT_PCT", 6.0))

    # ---- backtest ----
    commission_per_trade: float = field(default_factory=lambda: _get_float("COMMISSION_PER_TRADE", 0.0))
    slippage_bps: float = field(default_factory=lambda: _get_float("SLIPPAGE_BPS", 5.0))

    # ---- execution scheduling (Almgren-Chriss; opt-in, default = legacy) ----
    execution_mode: str = field(default_factory=lambda: _get("EXECUTION_MODE", "naive"))  # naive | twap | ac
    exec_horizon_s: float = field(default_factory=lambda: _get_float("EXEC_HORIZON_S", 600.0))
    exec_slices: int = field(default_factory=lambda: _get_int("EXEC_SLICES", 10))

    def assert_paper_safe(self) -> None:
        """Refuse to run against anything that looks like a live venue unless
        the user explicitly enabled live trading."""
        # NB: match on "://<host>" so the paper host
        # "paper-api.alpaca.markets" does NOT trip the live marker
        # "api.alpaca.markets" (bare substring would false-positive on paper).
        live_markers = ("://api.alpaca.markets", "://www.deribit.com")
        if any(m in self.alpaca_base_url for m in live_markers) or (
            not self.deribit_testnet and self.mode == "paper"
        ):
            if not self.live_trading_enabled:
                raise RuntimeError(
                    "Refusing to run: target looks like a LIVE venue but "
                    "LIVE_TRADING_ENABLED is not 'true'. Paper trading only."
                )
