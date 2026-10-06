"""Walk-forward splits by trading session with an embargo.

Labels never cross a session boundary (positions are flat by 15:55), so purging
reduces to: training sessions strictly before the test block, minus `embargo_days`
sessions to kill serial correlation in multi-day features (rel_vol, ret_5d)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WalkForwardConfig


@dataclass
class Fold:
    k: int
    train_days: list
    val_days: list      # time-ordered tail of the training window (early stop + thresholds)
    test_days: list


def walk_forward(days: list, cfg: WalkForwardConfig, inner_val_frac: float) -> list[Fold]:
    days = sorted(days)
    folds, k = [], 0
    start = cfg.min_train_days + cfg.embargo_days
    while start < len(days):
        test = days[start:start + cfg.test_days]
        train_end = start - cfg.embargo_days
        train_start = 0 if cfg.expanding else max(0, train_end - cfg.min_train_days)
        window = days[train_start:train_end]
        n_val = max(3, int(round(len(window) * inner_val_frac)))
        folds.append(Fold(k, window[:-n_val], window[-n_val:], test))
        start += cfg.test_days
        k += 1
    return folds


def assert_no_overlap(f: Fold):
    tr, va, te = set(f.train_days), set(f.val_days), set(f.test_days)
    assert not (tr & te) and not (va & te) and not (tr & va)
    assert max(f.train_days) < min(f.val_days) < min(f.test_days), "time order violated"


def percentile_rank(reference: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rank of x within a reference distribution that was available at the time."""
    ref = np.sort(reference)
    return np.searchsorted(ref, x, side="right") / max(len(ref), 1)
