"""Broker abstraction. Strategy + RiskManager talk only to `Broker`, so backtest,
paper and live share one code path. AlpacaBroker needs `pip install alpaca-py`
and API keys in env vars (ALPACA_KEY_ID / ALPACA_SECRET_KEY). Default is PAPER."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BracketOrder:
    symbol: str
    side: str            # "buy" | "sell" (sell = short entry)
    qty: int
    take_profit: float
    stop_loss: float
    client_id: str       # idempotency key: f"{date}-{symbol}-{bar_ts}" -> no duplicate orders on retry


class Broker(ABC):
    @abstractmethod
    def submit_bracket(self, o: BracketOrder) -> str: ...
    @abstractmethod
    def positions(self) -> dict: ...
    @abstractmethod
    def cancel_all_and_flatten(self) -> None: ...
    @abstractmethod
    def account_equity(self) -> float: ...


class AlpacaBroker(Broker):
    def __init__(self, paper: bool = True):
        from alpaca.trading.client import TradingClient
        if not paper and os.environ.get("EDGELAB_LIVE_CONFIRM") != "I_ACCEPT_REAL_MONEY_RISK":
            raise RuntimeError("Live trading requires EDGELAB_LIVE_CONFIRM to be set explicitly.")
        self.client = TradingClient(os.environ["ALPACA_KEY_ID"], os.environ["ALPACA_SECRET_KEY"], paper=paper)

    def submit_bracket(self, o: BracketOrder) -> str:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
        req = MarketOrderRequest(
            symbol=o.symbol, qty=o.qty, side=OrderSide.BUY if o.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET, client_order_id=o.client_id,
            take_profit=TakeProfitRequest(limit_price=round(o.take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(o.stop_loss, 2)))
        return str(self.client.submit_order(req).id)

    def positions(self) -> dict:
        return {p.symbol: int(float(p.qty)) for p in self.client.get_all_positions()}

    def cancel_all_and_flatten(self) -> None:
        self.client.close_all_positions(cancel_orders=True)

    def account_equity(self) -> float:
        return float(self.client.get_account().equity)


class SimBroker(Broker):
    """In-memory broker for dry runs / unit tests: records orders, fills nothing."""
    def __init__(self, equity: float):
        self.orders, self._equity = [], equity

    def submit_bracket(self, o: BracketOrder) -> str:
        if any(x.client_id == o.client_id for x in self.orders):
            return o.client_id
        self.orders.append(o)
        return o.client_id

    def positions(self) -> dict:
        return {o.symbol: o.qty for o in self.orders}

    def cancel_all_and_flatten(self) -> None:
        self.orders.clear()

    def account_equity(self) -> float:
        return self._equity
