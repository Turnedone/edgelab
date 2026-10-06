"""Performance + statistical-significance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


def sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(np.sqrt(ANN) * r.mean() / sd) if sd > 0 else 0.0


def sortino(r: pd.Series) -> float:
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    return float(np.sqrt(ANN) * r.mean() / dd) if dd > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def cagr(equity: pd.Series, start: float) -> float:
    n = len(equity)
    return float((equity.iloc[-1] / start) ** (ANN / n) - 1) if n else 0.0


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark) accounting for sample length, skew and kurtosis
    (Bailey & Lopez de Prado 2012). Uses per-period (non-annualised) Sharpe."""
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return 0.5
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sr_std: float | None = None) -> float:
    """PSR against the Sharpe you'd expect from the BEST of `n_trials` useless
    strategies. If you tried 50 configs, a lucky one will look good: this deflates it."""
    n = len(r)
    if n < 3:
        return 0.0
    sd = trial_sr_std if trial_sr_std is not None else 1 / np.sqrt(n - 1)
    emc = 0.5772156649
    z = (1 - emc) * stats.norm.ppf(1 - 1 / max(n_trials, 2)) + emc * stats.norm.ppf(1 - 1 / (max(n_trials, 2) * np.e))
    return probabilistic_sharpe(r, sr_benchmark=sd * z)


def block_bootstrap_ci(r: pd.Series, fn=sharpe, block: int = 5, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = r.to_numpy()
    if len(x) < block * 2:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(n):
        starts = rng.integers(0, len(x) - block + 1, int(np.ceil(len(x) / block)))
        sample = np.concatenate([x[s:s + block] for s in starts])[:len(x)]
        out.append(fn(pd.Series(sample)))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def trade_stats(tr: pd.DataFrame) -> dict:
    if not len(tr):
        return {"trades": 0}
    wins, losses = tr.loc[tr.pnl > 0, "pnl"], tr.loc[tr.pnl <= 0, "pnl"]
    t, p = stats.ttest_1samp(tr["net_ret"], 0.0) if len(tr) > 2 else (np.nan, np.nan)
    return {
        "trades": int(len(tr)),
        "win_rate": float((tr.pnl > 0).mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "avg_net_bps": float(tr["net_ret"].mean() * 1e4),
        "avg_gross_bps": float(tr["gross_ret"].mean() * 1e4),
        "avg_cost_bps": float((tr["cost"] / tr["notional"]).mean() * 1e4),
        "avg_r": float(tr["r_multiple"].mean()),
        "t_stat_trades": float(t), "p_value_trades": float(p),
        "pct_tp": float((tr.reason == 1).mean()), "pct_sl": float((tr.reason == 2).mean()),
        "pct_time": float((tr.reason == 3).mean()),
    }


def summary(bt: dict, start_equity: float, n_trials: int) -> dict:
    r, eq, tr = bt["daily_ret"], bt["equity"], bt["trades"]
    turnover = float(tr["notional"].sum() * 2 / eq.mean() / max(len(eq), 1)) if len(tr) else 0.0
    out = {
        "days": int(len(r)), "total_return": float(eq.iloc[-1] / start_equity - 1),
        "cagr_extrapolated": cagr(eq, start_equity), "sharpe": sharpe(r), "sortino": sortino(r),
        "max_drawdown": max_drawdown(pd.concat([pd.Series([start_equity]), eq])),
        "daily_turnover_x": turnover, "psr_vs_0": probabilistic_sharpe(r),
        "deflated_sharpe": deflated_sharpe(r, n_trials), "sharpe_ci95": block_bootstrap_ci(r),
        "pct_days_traded": float((bt["daily_pnl"] != 0).mean()),
    }
    out.update(trade_stats(tr))
    return out
