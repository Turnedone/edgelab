"""Paper-trading loop (Stage 4). Runs one decision cycle per completed 5m bar.

    python scripts/paper_trade.py --dry-run      # SimBroker, prints intended orders
    python scripts/paper_trade.py                # Alpaca PAPER account

Every decision (features hash, p_long, p_short, risk verdict, order id) is appended
to logs/decisions.jsonl so live behaviour can be compared to the backtest later.
Production note: for real use replace the yfinance poll with a streaming feed
(Alpaca SIP websocket / Polygon) — yfinance is delayed and rate-limited."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edgelab.config import CONTEXT, UNIVERSE, Config  # noqa: E402
from edgelab.data import ET, fetch_yfinance  # noqa: E402
from edgelab.execution.broker import AlpacaBroker, BracketOrder, SimBroker  # noqa: E402
from edgelab.features import FEATURES, build_features  # noqa: E402
from edgelab.model import ProbModel  # noqa: E402
from edgelab.research import build_dataset  # noqa: E402
from edgelab.risk import RiskManager  # noqa: E402
from edgelab.validation import percentile_rank  # noqa: E402

LOG = ROOT / "logs" / "decisions.jsonl"


def train_production_models(cfg: Config):
    """Fit on ALL history except a recent validation tail (used for thresholds)."""
    _, ds = build_dataset(cfg, refresh=True)
    days = sorted(ds["date"].unique())
    n_val = max(3, int(len(days) * cfg.model.inner_val_frac))
    tr, va = ds[ds.date.isin(days[:-n_val])], ds[ds.date.isin(days[-n_val:])]
    out = {}
    for side in ("long", "short"):
        m = ProbModel(cfg.model).fit(tr[FEATURES], tr[f"y_{side}"].to_numpy(), va[FEATURES], va[f"y_{side}"].to_numpy())
        out[side] = (m, m.predict_proba(va[FEATURES]))
    return out


def reconcile(rm: RiskManager, broker, day_start_equity: float, now):
    """Broker is the source of truth: release closed positions, sync equity and the
    day's realised P&L so the daily loss limit sees real fills, not model guesses."""
    held = broker.positions()
    eq = broker.account_equity()
    for sym in [s for s in rm.open if s not in held]:
        pos = rm.open[sym]
        rm.on_close(pos, 0.0, now, cfg_cooldown(rm))
    rm.equity = eq
    rm.realized_today = eq - day_start_equity
    rm.peak_equity = max(rm.peak_equity, eq)
    if rm.realized_today <= -rm.cfg.daily_loss_limit * eq:
        rm.halted_today = True


def cfg_cooldown(rm):
    return rm.cfg.cooldown_bars_after_loss * pd.Timedelta(minutes=5)


def cycle(cfg, models, rm: RiskManager, broker, now):
    raw = fetch_yfinance(UNIVERSE + CONTEXT, cfg.interval, "15d")
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True).dt.tz_convert(ET)
    raw["date"] = raw["ts"].dt.date
    raw = raw[raw["ts"] + pd.Timedelta(minutes=5) <= now]          # completed bars only
    feats = build_features(raw.sort_values(["symbol", "ts"]).reset_index(drop=True), UNIVERSE)
    feats = feats[feats["ts"] == feats["ts"].max()].reset_index(drop=True)
    b = np.clip(cfg.labels.vol_mult * feats["sigma_30m"], cfg.labels.min_barrier, cfg.labels.max_barrier)
    for side in ("long", "short"):
        m, ref = models[side]
        feats[f"p_{side}"] = m.predict_proba(feats[FEATURES])
        feats[f"pq_{side}"] = percentile_rank(ref, feats[f"p_{side}"].to_numpy())
    feats["barrier"] = b
    q = cfg.wf.signal_quantile
    be = (b + 4.5e-4) / (2 * b)
    LOG.parent.mkdir(exist_ok=True)
    for r in feats.itertuples():
        d = 1 if (r.pq_long >= q and r.p_long > be[r.Index] and r.p_long >= r.p_short) else \
            -1 if (r.pq_short >= q and r.p_short > be[r.Index]) else 0
        verdict, oid = "no_signal", None
        if d:
            px = float(r.close)
            pos = rm.size_and_approve(r.symbol, d, px, float(r.barrier), now)
            if pos is None:
                verdict = "risk_rejected"
            else:
                tp = px * (1 + d * r.barrier)
                sl = px * (1 - d * r.barrier)
                oid = broker.submit_bracket(BracketOrder(r.symbol, "buy" if d == 1 else "sell", pos.shares,
                                                         tp, sl, f"{now:%Y%m%d%H%M}-{r.symbol}"))
                verdict = "submitted"
        with LOG.open("a") as fh:
            fh.write(json.dumps({"ts": str(now), "symbol": r.symbol, "p_long": float(r.p_long),
                                 "p_short": float(r.p_short), "dir": d, "verdict": verdict, "order": oid}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = Config()
    models = train_production_models(cfg)
    broker = SimBroker(cfg.start_equity) if a.dry_run else AlpacaBroker(paper=True)
    rm = RiskManager(cfg.risk, broker.account_equity())
    day_start = rm.equity
    flatten = dt.time.fromisoformat(cfg.risk.flatten_at)
    while True:
        now = pd.Timestamp.now(tz=ET)
        if now.time() >= flatten:
            broker.cancel_all_and_flatten()
            print("Flat for the day.")
            break
        if now.time() >= dt.time(9, 45) and now.minute % 5 == 0 and now.second < 20:
            reconcile(rm, broker, day_start, now)
            cycle(cfg, models, rm, broker, now.floor("5min"))
            time.sleep(30)
        time.sleep(5)


if __name__ == "__main__":
    main()
