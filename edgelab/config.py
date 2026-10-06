"""Central configuration. Every number that affects results lives here so that
it is versioned with the results it produced (see results/report.json)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# 25 highly liquid US large caps + QQQ. Sector ETFs are context-only (never traded).
UNIVERSE = [
    "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "AMD",
    "NFLX", "COST", "ADBE", "CRM", "QCOM", "MU", "AMAT", "CSCO", "PEP", "TXN",
    "JPM", "XOM", "UNH", "V", "MA", "INTC",
]
CONTEXT = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLC", "SMH", "XLP"]
SECTOR = {
    "QQQ": "XLK", "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SMH", "AVGO": "SMH", "AMD": "SMH",
    "QCOM": "SMH", "MU": "SMH", "AMAT": "SMH", "TXN": "SMH", "INTC": "SMH", "ADBE": "XLK",
    "CRM": "XLK", "CSCO": "XLK", "AMZN": "XLY", "TSLA": "XLY", "COST": "XLP", "PEP": "XLP",
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC", "JPM": "XLF", "V": "XLF", "MA": "XLF",
    "XOM": "XLE", "UNH": "XLV",
}


@dataclass
class LabelConfig:
    horizon_bars: int = 6            # 6 x 5m = 30 minutes
    barrier_mode: str = "vol"        # "vol" (recommended) or "fixed"
    fixed_barrier: float = 0.004     # +/-0.4% if barrier_mode == "fixed"
    vol_mult: float = 1.0            # barrier = vol_mult * sigma_30min
    min_barrier: float = 0.0025
    max_barrier: float = 0.0100
    tp_touch_buffer_bps: float = 1.0  # price must trade THROUGH the limit to count as filled


@dataclass
class CostConfig:
    half_spread_bps: float = 1.0     # megacaps: ~0.5-2bp half-spread; QQQ ~0.3bp
    slippage_bps: float = 1.0        # market-order slippage per side
    stop_extra_slippage_bps: float = 2.0  # stops fill worse than limits
    latency_bps: float = 0.5         # adverse drift during signal->order latency
    commission_per_share: float = 0.0035  # IBKR-tiered-like; Alpaca is 0 but keep conservative
    min_commission: float = 0.35
    sec_fee_per_dollar_sold: float = 27.80e-6
    finra_taf_per_share: float = 0.000166
    stress_multiplier: float = 1.0   # robustness runs use 2x and 3x


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.0035   # 0.35% of equity at stop
    max_position_notional: float = 0.25  # of equity
    max_gross_exposure: float = 1.0  # no leverage
    max_concurrent: int = 4
    max_trades_per_day: int = 12
    daily_loss_limit: float = 0.015  # stop opening trades at -1.5% (incl. open risk)
    max_drawdown_kill: float = 0.08  # hard kill switch -> human review
    cooldown_bars_after_loss: int = 6  # no revenge re-entry on same symbol for 30m
    no_entry_first_minutes: int = 15
    no_entry_after: str = "15:20"
    flatten_at: str = "15:55"


@dataclass
class ModelConfig:
    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 4
    num_leaves: int = 15
    min_child_samples: int = 200     # strong regularisation: intraday labels are noisy
    subsample: float = 0.7
    colsample: float = 0.7
    l2: float = 5.0
    inner_val_frac: float = 0.2      # tail of each training window for early stopping/thresholds
    seed: int = 7


@dataclass
class WalkForwardConfig:
    min_train_days: int = 20
    test_days: int = 5
    embargo_days: int = 1
    expanding: bool = True
    signal_quantile: float = 0.95    # trade only the top 5% of p, quantile set on TRAIN-period data


@dataclass
class Config:
    interval: str = "5m"
    period: str = "60d"
    start_equity: float = 25_000.0
    labels: LabelConfig = field(default_factory=LabelConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wf: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    n_configs_tried: int = 6         # honest count of variants you looked at -> Deflated Sharpe

    def to_dict(self) -> dict:
        return asdict(self)
