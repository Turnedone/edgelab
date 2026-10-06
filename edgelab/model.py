"""Gradient-boosted probability model. LightGBM if it loads, otherwise sklearn's
HistGradientBoostingClassifier (same algorithm family: histogram GBDT)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelConfig

try:  # LightGBM's DLL fails to load on some Windows paths; degrade gracefully.
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:  # noqa: BLE001
    HAVE_LGB = False
from sklearn.ensemble import HistGradientBoostingClassifier


class ProbModel:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = None
        self.features: list[str] = []
        self.backend = "lightgbm" if HAVE_LGB else "sklearn-hgb"

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray):
        c = self.cfg
        self.features = list(X.columns)
        if HAVE_LGB:
            self.model = lgb.LGBMClassifier(
                n_estimators=c.n_estimators, learning_rate=c.learning_rate, max_depth=c.max_depth,
                num_leaves=c.num_leaves, min_child_samples=c.min_child_samples,
                subsample=c.subsample, subsample_freq=1, colsample_bytree=c.colsample,
                reg_lambda=c.l2, random_state=c.seed, verbose=-1)
            self.model.fit(X, y, eval_set=[(X_val, y_val)],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            # HGB's early stopping uses a random split -> leaks time order. Instead we
            # pick the iteration count on the time-ordered validation tail ourselves.
            m = HistGradientBoostingClassifier(
                max_iter=c.n_estimators, learning_rate=c.learning_rate, max_depth=c.max_depth,
                max_leaf_nodes=c.num_leaves, min_samples_leaf=c.min_child_samples,
                l2_regularization=c.l2, max_features=c.colsample, early_stopping=False,
                random_state=c.seed)
            m.fit(X, y)
            losses = []
            for p in m.staged_predict_proba(X_val):
                p1 = np.clip(p[:, 1], 1e-6, 1 - 1e-6)
                losses.append(-np.mean(y_val * np.log(p1) + (1 - y_val) * np.log(1 - p1)))
            best = int(np.argmin(losses)) + 1
            m.set_params(max_iter=max(best, 10))
            m.fit(X, y)
            self.model = m
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def importance(self, X_val: pd.DataFrame, y_val: np.ndarray, n_repeats: int = 3) -> pd.Series:
        """Permutation importance on held-out data (log-loss increase). Split-gain
        importances are biased toward high-cardinality features; this is not."""
        rng = np.random.default_rng(self.cfg.seed)
        base = _logloss(y_val, self.predict_proba(X_val))
        imp = {}
        for f in self.features:
            deltas = []
            for _ in range(n_repeats):
                Xp = X_val.copy()
                Xp[f] = rng.permutation(Xp[f].to_numpy())
                deltas.append(_logloss(y_val, self.predict_proba(Xp)) - base)
            imp[f] = float(np.mean(deltas))
        return pd.Series(imp).sort_values(ascending=False)


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
