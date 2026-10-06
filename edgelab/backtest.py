"""Event-driven portfolio backtest over pre-computed trade candidates.

Each candidate already carries its gross outcome from labels.py (decided only by
bars AFTER the signal). This module adds what the label cannot know: portfolio
constraints, sizing on current equity, the daily loss limit, cooldowns and costs.
Event ordering is pessimistic: an entry at the open of bar j is processed after
exits that complete at the end of bar j-1 and before exits during bar j."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .costs import round_trip_cost
from .risk import RiskManager

BAR = pd.Timedelta(minutes=5)


def select_signals(oos: pd.DataFrame, quantile: float, cfg: Config) -> pd.DataFrame:
    """Deterministic decision rule on top of model probabilities:
    1. probability must be in the top (1-quantile) of what the model produced on
       its own TRAINING-period validation slice (pq_* computed causally per fold);
    2. calibrated probability must clear the cost-aware break-even: p > (b + c)/(2b);
    3. take the stronger side if both qualify."""
    c_frac = 4.5 / 1e4 * cfg.costs.stress_multiplier
    be = (oos["barrier"] + c_frac) / (2 * oos["barrier"])
    long_ok = (oos["pq_long"] >= quantile) & (oos["p_long"] > be)
    short_ok = (oos["pq_short"] >= quantile) & (oos["p_short"] > be)
    pick_long = long_ok & (~short_ok | (oos["p_long"] >= oos["p_short"]))
    pick_short = short_ok & ~pick_long
    s = pd.concat([_side(oos[pick_long], 1), _side(oos[pick_short], -1)])
    return s.sort_values(["entry_ts", "p"], ascending=[True, False]).reset_index(drop=True)


def _side(d: pd.DataFrame, direction: int) -> pd.DataFrame:
    n = "long" if direction == 1 else "short"
    return pd.DataFrame({
        "ts": d["ts"], "entry_ts": d["entry_ts"], "exit_ts": d[f"exit_ts_{n}"],
        "symbol": d["symbol"], "direction": direction, "entry_px": d["entry_px"],
        "barrier": d["barrier"], "gross": d[f"gross_{n}"], "reason": d[f"reason_{n}"],
        "p": d[f"p_{n}"], "y": d[f"y_{n}"], "fold": d["fold"],
    })


def run_backtest(signals: pd.DataFrame, cfg: Config, test_days: list) -> dict:
    rm = RiskManager(cfg.risk, cfg.start_equity)
    cooldown = cfg.risk.cooldown_bars_after_loss * BAR
    events = []
    for i, s in enumerate(signals.itertuples(index=False)):
        events.append((s.entry_ts, 1, i))
        events.append((s.exit_ts + BAR, 0, i))   # exit completes at end of exit bar
    events.sort(key=lambda e: (e[0], e[1]))
    live, trades = {}, []
    for t, kind, i in events:
        s = signals.iloc[i]
        if kind == 1:
            pos = rm.size_and_approve(s.symbol, int(s.direction), float(s.entry_px), float(s.barrier), t)
            if pos is not None:
                live[i] = (pos, rm.equity)
        elif i in live:
            pos, eq_at_entry = live.pop(i)
            cost = round_trip_cost(cfg.costs, pos.entry_px, pos.shares, int(s.reason))
            gross_pnl = pos.notional * float(s.gross)
            pnl = gross_pnl - cost
            rm.on_close(pos, pnl, t, cooldown)
            trades.append({
                "symbol": s.symbol, "direction": int(s.direction), "entry_ts": s.entry_ts,
                "exit_ts": s.exit_ts, "p": float(s.p), "shares": pos.shares,
                "notional": pos.notional, "gross_ret": float(s.gross), "reason": int(s.reason),
                "gross_pnl": gross_pnl, "cost": cost, "pnl": pnl,
                "net_ret": pnl / pos.notional, "r_multiple": pnl / pos.risk_dollars,
                "equity_after": rm.equity, "fold": int(s.fold),
            })
    tr = pd.DataFrame(trades)
    daily = pd.Series(0.0, index=pd.Index(sorted(test_days), name="date"))
    if len(tr):
        tr["date"] = pd.to_datetime(tr["exit_ts"]).dt.date
        daily = daily.add(tr.groupby("date")["pnl"].sum(), fill_value=0.0)
    equity = cfg.start_equity + daily.cumsum()
    daily_ret = daily / equity.shift(1).fillna(cfg.start_equity)
    return {"trades": tr, "daily_pnl": daily, "daily_ret": daily_ret, "equity": equity,
            "rejections": dict(rm.rejections), "killed": rm.killed}


def random_entry_baseline(oos: pd.DataFrame, n_trades: int, n_iter: int, cost_frac: float,
                          seed: int = 0) -> np.ndarray:
    """Distribution of mean net trade return for RANDOM entries drawn from the same
    OOS rows (same universe, same hours, random side). If the strategy's mean is not
    in the far right tail of this, the model adds nothing."""
    rng = np.random.default_rng(seed)
    gl, gs = oos["gross_long"].to_numpy(), oos["gross_short"].to_numpy()
    means = np.empty(n_iter)
    for k in range(n_iter):
        idx = rng.integers(0, len(oos), n_trades)
        side = rng.random(n_trades) < 0.5
        means[k] = np.mean(np.where(side, gl[idx], gs[idx])) - cost_frac
    return means
