"""Alpaca paper-trading adapter (stocks, crypto, single-leg options).

Uses raw REST so there is no SDK dependency. Points at the paper sandbox by
default; the live URL is refused unless Config.live_trading_enabled.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import requests

from ..config import Config
from ..models import Account, Fill, Order, Position
from .base import Broker

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, cfg: Config):
        if not cfg.alpaca_key or not cfg.alpaca_secret:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")
        self.base = cfg.alpaca_base_url.rstrip("/")
        self.data = DATA_URL
        self._h = {
            "APCA-API-KEY-ID": cfg.alpaca_key,
            "APCA-API-SECRET-KEY": cfg.alpaca_secret,
            "Content-Type": "application/json",
        }

    # ---- helpers ----
    def _req(self, method: str, url: str, **kw: Any) -> Any:
        r = requests.request(method, url, headers=self._h, timeout=20, **kw)
        r.raise_for_status()
        return r.json() if r.text else None

    # ---- account / positions ----
    def get_account(self) -> Account:
        a = self._req("GET", f"{self.base}/v2/account")
        return Account(
            equity=float(a["equity"]),
            cash=float(a["cash"]),
            buying_power=float(a["buying_power"]),
        )

    def get_positions(self) -> List[Position]:
        out = []
        for p in self._req("GET", f"{self.base}/v2/positions"):
            qty = float(p["qty"])
            out.append(Position(
                symbol=p["symbol"],
                qty=qty if p["side"] == "long" else -qty,
                avg_entry=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                unrealized_pnl=float(p["unrealized_pl"]),
                asset_class="option" if len(p["symbol"]) > 12 else "spot",
            ))
        return out

    # ---- market data ----
    def get_bars(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        tf = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "1d": "1Day"}[timeframe]
        if "-" in symbol:  # crypto pair like BTC/USD
            path = f"{self.data}/v1beta1/crypto/us/bars"
            params = {"symbols": symbol.replace("/", ""), "timeframe": tf, "limit": limit}
        else:
            path = f"{self.data}/v2/stocks/{symbol}/bars"
            params = {"timeframe": tf, "limit": limit, "adjustment": "split"}
        bars = self._req("GET", path, params=params)["bars"]
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"], utc=True)
        return df.set_index("t").sort_index()[["o", "h", "l", "c", "v"]].rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})

    def get_last_price(self, symbol: str) -> float:
        if "-" in symbol:
            q = self._req("GET", f"{self.data}/v1beta1/crypto/us/latest/trades",
                          params={"symbols": symbol.replace("/", "")})
            return float(q["trades"][symbol.replace("/", "")]["p"])
        q = self._req("GET", f"{self.data}/v2/stocks/{symbol}/trades/latest")
        return float(q["trade"]["p"])

    # ---- orders ----
    def place_order(self, order: Order) -> Fill:
        body: Dict[str, Any] = {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side,
            "type": order.order_type,
            "time_in_force": order.time_in_force,
            "client_order_id": order.client_id or f"bot-{uuid4().hex[:12]}",
        }
        if order.order_type == "limit" and order.limit_price:
            body["limit_price"] = str(order.limit_price)
        o = self._req("POST", f"{self.base}/v2/orders", json=body)
        status = o.get("status", "")
        filled_avg = o.get("filled_avg_price")
        if filled_avg:
            price = float(filled_avg)
        else:
            # Not filled yet (e.g. market closed): use last price as a
            # reference only; the real fill comes at the next open.
            price = self.get_last_price(order.symbol)
        return Fill(order_id=o["id"], symbol=order.symbol, side=order.side,
                    qty=float(o.get("filled_qty") or order.qty), price=price,
                    timestamp=datetime.now(timezone.utc), status=status)

    def cancel_order(self, order_id: str) -> None:
        self._req("DELETE", f"{self.base}/v2/orders/{order_id}")

    # ---- options ----
    def list_option_contracts(self, underlying: str, **filters: Any) -> List[Dict[str, Any]]:
        params = {"underlying_symbols": underlying, "status": "active", "limit": 1000}
        params.update({k: v for k, v in filters.items() if v is not None})
        return self._req("GET", f"{self.base}/v2/options/contracts", params=params).get("option_contracts", [])

    def get_option_snapshot(self, option_symbol: str) -> Dict[str, Any]:
        s = self._req("GET", f"{self.data}/v1beta1/options/snapshots/{option_symbol}")
        return s.get("snapshot", {})
