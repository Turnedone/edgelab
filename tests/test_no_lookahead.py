"""Leakage test: corrupt every bar AFTER a cutoff and assert no feature at or before
the cutoff changes. Run:  python -m pytest tests -q   (or python tests/test_no_lookahead.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edgelab.config import CONTEXT, UNIVERSE, WalkForwardConfig  # noqa: E402
from edgelab.data import load_bars  # noqa: E402
from edgelab.features import FEATURES, build_features  # noqa: E402
from edgelab.validation import assert_no_overlap, walk_forward  # noqa: E402


def test_features_are_point_in_time():
    bars = load_bars(UNIVERSE + CONTEXT, refresh=False)
    days = sorted(bars["date"].unique())
    cut = pd.Timestamp(f"{days[len(days) // 2]} 12:00", tz="America/New_York")
    a = build_features(bars, UNIVERSE)
    bad = bars.copy()
    fut = bad["ts"] > cut
    rng = np.random.default_rng(0)
    for col in ("open", "high", "low", "close"):
        bad.loc[fut, col] *= rng.uniform(0.5, 1.5, fut.sum())
    bad.loc[fut, "volume"] *= 7
    b = build_features(bad, UNIVERSE)
    ka, kb = a[a.ts <= cut].set_index(["ts", "symbol"]), b[b.ts <= cut].set_index(["ts", "symbol"])
    diff = (ka[FEATURES] - kb[FEATURES]).abs().max()
    assert (diff.fillna(0) < 1e-9).all(), f"look-ahead in: {diff[diff > 1e-9].to_dict()}"


def test_walk_forward_is_time_ordered():
    for f in walk_forward(list(range(60)), WalkForwardConfig(), 0.2):
        assert_no_overlap(f)
        assert min(f.test_days) - max(f.val_days) > 1   # embargo respected


if __name__ == "__main__":
    test_walk_forward_is_time_ordered()
    test_features_are_point_in_time()
    print("no look-ahead: OK")
