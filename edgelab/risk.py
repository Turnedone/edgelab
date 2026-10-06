"""Deterministic risk layer. The SAME class is used by the backtester, the paper
trader and the live trader, so a rule cannot exist in one and not the others.
The ML model only proposes; this layer disposes."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .config import RiskConfig


@dataclass
class Position:
    symbol: str
    direction: int
    shares: int
    entry_px: float
    stop_dist: float          # fraction of price
    entry_ts: object
    exit_ts: object = None

    @property
    def risk_dollars(self) -> float:
        return self.shares * self.entry_px * self.stop_dist

    @property
    def notional(self) -> float:
        return self.shares * self.entry_px


@dataclass
class RiskManager:
    cfg: RiskConfig
    equity: float
    peak_equity: float = 0.0
    day: object = None
    realized_today: float = 0.0
    trades_today: int = 0
    open: dict = field(default_factory=dict)
    cooldown_until: dict = field(default_factory=dict)
    halted_today: bool = False
    killed: bool = False
    rejections: dict = field(default_factory=dict)

    def __post_init__(self):
        self.peak_equity = self.peak_equity or self.equity

    # ---- lifecycle -------------------------------------------------------
    def new_day(self, day):
        if day != self.day:
            self.day, self.realized_today, self.trades_today, self.halted_today = day, 0.0, 0, False

    def _reject(self, why: str):
        self.rejections[why] = self.rejections.get(why, 0) + 1
        return None

    # ---- decision --------------------------------------------------------
    def size_and_approve(self, symbol: str, direction: int, price: float, stop_dist: float, ts) -> Position | None:
        c = self.cfg
        self.new_day(ts.date())
        if self.killed:
            return self._reject("kill_switch")
        if self.halted_today:
            return self._reject("daily_loss_limit")
        t = ts.time()
        open_t = dt.time(9, 30)
        mins = (ts.hour - open_t.hour) * 60 + ts.minute - open_t.minute
        if mins < c.no_entry_first_minutes or t > dt.time.fromisoformat(c.no_entry_after):
            return self._reject("time_window")
        if symbol in self.open:
            return self._reject("already_in_symbol")
        if len(self.open) >= c.max_concurrent:
            return self._reject("max_concurrent")
        if self.trades_today >= c.max_trades_per_day:
            return self._reject("max_trades_day")
        if self.cooldown_until.get(symbol) is not None and ts < self.cooldown_until[symbol]:
            return self._reject("cooldown_after_loss")
        open_risk = sum(p.risk_dollars for p in self.open.values())
        risk_dollars = self.equity * c.risk_per_trade
        # Worst case if every open stop + this one is hit must stay inside the daily limit.
        if self.realized_today - open_risk - risk_dollars < -c.daily_loss_limit * self.equity:
            return self._reject("daily_limit_headroom")
        shares = int(risk_dollars / (price * stop_dist))
        shares = min(shares, int(c.max_position_notional * self.equity / price))
        gross = sum(p.notional for p in self.open.values())
        shares = min(shares, int(max(0.0, c.max_gross_exposure * self.equity - gross) / price))
        if shares < 1:
            return self._reject("size_zero")
        pos = Position(symbol, direction, shares, price, stop_dist, ts)
        self.open[symbol] = pos
        self.trades_today += 1
        return pos

    def on_close(self, pos: Position, pnl: float, ts, cooldown=None):
        self.new_day(ts.date())
        self.open.pop(pos.symbol, None)
        self.equity += pnl
        self.realized_today += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        if pnl < 0 and cooldown is not None:
            self.cooldown_until[pos.symbol] = ts + cooldown
        if self.realized_today <= -self.cfg.daily_loss_limit * self.equity:
            self.halted_today = True
        if self.equity <= (1 - self.cfg.max_drawdown_kill) * self.peak_equity:
            self.killed = True   # requires a human to reset: no automatic restart
