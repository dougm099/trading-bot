"""Deribit testnet adapter (crypto perps, futures, options) via raw REST.

Auth uses OAuth2 client-credentials against /public/auth. Public market data
works without keys. Testnet URL: https://test.deribit.com/api/v2
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import requests

from ..config import Config
from ..models import Account, Fill, Order, Position
from .base import Broker

TESTNET = "https://test.deribit.com/api/v2"
MAINNET = "https://www.deribit.com/api/v2"


class DeribitBroker(Broker):
    name = "deribit"

    def __init__(self, cfg: Config):
        self.base = (TESTNET if cfg.deribit_testnet else MAINNET)
        self.cid = cfg.deribit_client_id
        self.secret = cfg.deribit_client_secret
        self._token: Optional[str] = None
        self._token_exp: float = 0.0

    # ---- http ----
    def _call(self, method: str, params: Optional[Dict[str, Any]] = None,
              private: bool = False) -> Any:
        headers: Dict[str, str] = {}
        if private:
            headers["Authorization"] = f"bearer {self._auth_token()}"
        r = requests.get(f"{self.base}/{method}", params=params or {},
                         headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"Deribit error: {data['error']}")
        return data["result"]

    def _auth_token(self) -> str:
        if not self.cid or not self.secret:
            raise RuntimeError("DERIBIT_CLIENT_ID / DERIBIT_CLIENT_SECRET are not set")
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        res = self._call("public/auth", {
            "grant_type": "client_credentials",
            "client_id": self.cid,
            "client_secret": self.secret,
        })
        self._token = res["access_token"]
        self._token_exp = time.time() + res["expires_in"]
        return self._token

    # ---- account / positions ----
    def get_account(self) -> Account:
        # testnet accounts are funded in BTC; report in USD terms via index price
        summaries = self._call("private/get_account_summary", {"currency": "BTC"}, private=True)
        btc = float(summaries["equity"])
        idx = float(self._call("public/get_index_price", {"index_name": "btc_usd"})["index_price"])
        equity_usd = btc * idx
        return Account(equity=equity_usd, cash=equity_usd, buying_power=equity_usd)

    def get_positions(self) -> List[Position]:
        out: List[Position] = []
        for cur in ("BTC", "ETH"):
            for p in self._call("private/get_positions", {"currency": cur}, private=True):
                if float(p["size"]) == 0:
                    continue
                out.append(Position(
                    symbol=p["instrument_name"],
                    qty=float(p["size"]) * (1 if p["direction"] == "buy" else -1),
                    avg_entry=float(p["average_price"]),
                    current_price=float(p["mark_price"]),
                    unrealized_pnl=float(p["total_profit_loss"]),
                    asset_class="option" if p["kind"] == "option" else ("perp" if "PERPETUAL" in p["instrument_name"] else "future"),
                ))
        return out

    # ---- market data ----
    def get_bars(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        res = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": "1D"}[timeframe]
        now = int(time.time() * 1000)
        raw = self._call("public/get_tradingview_chart_data", {
            "instrument_name": symbol, "start_timestamp": now - limit * 3_600_000,
            "end_timestamp": now, "resolution": res,
        })
        df = pd.DataFrame({
            "t": pd.to_datetime(raw["ticks"], unit="ms", utc=True),
            "open": raw["open"], "high": raw["high"], "low": raw["low"],
            "close": raw["close"], "volume": raw["volume"],
        }).set_index("t").sort_index()
        return df.tail(limit)

    def get_last_price(self, symbol: str) -> float:
        t = self._call("public/ticker", {"instrument_name": symbol})
        return float(t["last_price"])

    # ---- orders ----
    def place_order(self, order: Order) -> Fill:
        verb = "buy" if order.side == "buy" else "sell"
        params: Dict[str, Any] = {
            "instrument_name": order.symbol,
            "amount": order.qty,
            "type": order.order_type,
            "label": order.client_id or f"bot-{uuid4().hex[:12]}",
        }
        if order.order_type == "limit" and order.limit_price:
            params["price"] = order.limit_price
        if order.reduce_only:
            params["reduce_only"] = True
        res = self._call(f"private/{verb}", params, private=True)["order"]
        price = float(res.get("average_price") or res.get("price") or 0.0) or self.get_last_price(order.symbol)
        return Fill(order_id=res["order_id"], symbol=order.symbol, side=order.side,
                    qty=float(res.get("filled_amount") or order.qty), price=price,
                    timestamp=datetime.now(timezone.utc))

    def cancel_order(self, order_id: str) -> None:
        self._call("private/cancel", {"order_id": order_id}, private=True)

    # ---- options helpers ----
    def list_option_contracts(self, underlying: str, **filters: Any) -> List[Dict[str, Any]]:
        cur = underlying.replace("PERPETUAL", "").split("-")[0]
        instruments = self._call("public/get_instruments", {"currency": cur, "kind": "option"})
        out = []
        for i in instruments:
            if filters.get("expiration") and not i["instrument_name"].startswith(filters["expiration"]):
                continue
            out.append({
                "symbol": i["instrument_name"],
                "strike": i["strike"],
                "option_type": i["option_type"],
                "expiration": i["expiration_timestamp"],
            })
        return out

    def get_option_snapshot(self, option_symbol: str) -> Dict[str, Any]:
        t = self._call("public/ticker", {"instrument_name": option_symbol})
        return {"bid": t.get("best_bid_price"), "ask": t.get("best_ask_price"),
                "mark": t.get("mark_price"), "iv": t.get("mark_iv"),
                "delta": t.get("greeks", {}).get("delta")}
