"""Transaction cost model. Every fill pays: half-spread + slippage + latency drift
(+ extra for stop fills) as price impact, plus explicit commissions/regulatory fees."""
from __future__ import annotations

from .config import CostConfig


def round_trip_cost(cfg: CostConfig, price: float, shares: int, exit_reason: int) -> float:
    """Dollar cost of entering and exiting `shares` at ~`price`.
    exit_reason: 1 take-profit (limit), 2 stop (market-ish), 3 time stop (market)."""
    m = cfg.stress_multiplier
    notional = price * shares
    entry_bps = cfg.half_spread_bps + cfg.slippage_bps + cfg.latency_bps
    if exit_reason == 1:      # passive limit: no spread crossing, but adverse selection is
        exit_bps = 0.0        # already priced in by requiring trade-through (labels.py)
    elif exit_reason == 2:
        exit_bps = cfg.half_spread_bps + cfg.slippage_bps + cfg.stop_extra_slippage_bps
    else:
        exit_bps = cfg.half_spread_bps + cfg.slippage_bps
    impact = notional * (entry_bps + exit_bps) * m / 1e4
    comm = 2 * max(cfg.min_commission, cfg.commission_per_share * shares)
    fees = notional * cfg.sec_fee_per_dollar_sold + min(8.30, cfg.finra_taf_per_share * shares)
    return impact + comm + fees


def cost_bps_estimate(cfg: CostConfig, exit_reason: int = 3) -> float:
    """Approximate round-trip cost in bps for a typical $5k clip of a $200 stock."""
    return round_trip_cost(cfg, 200.0, 25, exit_reason) / 5000 * 1e4
