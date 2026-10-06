"""Usage:  python scripts/run_research.py [--no-refresh] [--no-variants]
Writes results/report.json, results/trades.csv, results/sim_data.json and
rebuilds dashboard/edgelab_dashboard.html."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edgelab.config import Config  # noqa: E402
from edgelab.research import export_candidates, run  # noqa: E402


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    return str(o)


def _clean(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--no-variants", action="store_true")
    a = ap.parse_args()
    cfg = Config()
    res = run(cfg, refresh=not a.no_refresh, variants=not a.no_variants)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rep = _clean(res["report"])
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=_json_default))
    res["trades"].to_csv(out / "trades.csv", index=False)

    oos = res["oos"]
    rng = np.random.default_rng(1)
    pool = oos.iloc[rng.choice(len(oos), min(4000, len(oos)), replace=False)].assign(pq_long=1.0, pq_short=1.0)
    sim = {"cols": ["e", "xl", "xs", "s", "px", "b", "pl", "ps", "ql", "qs", "gl", "gs", "rl", "rs"],
           "cand": export_candidates(oos), "pool": export_candidates(pool, 0.0),
           "epoch": "2000-01-01T00:00:00-05:00 (minutes, America/New_York wall clock)"}
    (out / "sim_data.json").write_text(json.dumps(sim, separators=(",", ":"), default=_json_default))

    s = rep["summary"]
    print("\n=== OUT-OF-SAMPLE RESULT (after costs) ===")
    for k in ("days", "trades", "total_return", "sharpe", "sortino", "max_drawdown", "profit_factor",
              "win_rate", "avg_gross_bps", "avg_cost_bps", "avg_net_bps", "p_value_trades",
              "psr_vs_0", "deflated_sharpe", "sharpe_ci95"):
        print(f"  {k:>18}: {s.get(k)}")
    print(f"  random-entry p-value: {rep['random_baseline']['p_value']:.3f}")

    from build_dashboard import build  # noqa: E402
    print("dashboard ->", build())


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
