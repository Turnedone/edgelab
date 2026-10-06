"""End-to-end research run: data -> features -> labels -> walk-forward models ->
OOS probabilities -> deterministic rules -> cost-aware backtest -> statistics."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from . import metrics as M
from .backtest import random_entry_baseline, run_backtest, select_signals
from .config import CONTEXT, UNIVERSE, Config
from .data import data_quality_report, load_bars
from .features import FEATURES, build_features
from .labels import make_labels
from .model import ProbModel
from .validation import assert_no_overlap, percentile_rank, walk_forward


def build_dataset(cfg: Config, refresh: bool = True):
    bars = load_bars(UNIVERSE + CONTEXT, cfg.interval, cfg.period, refresh=refresh)
    feats = build_features(bars, UNIVERSE)
    ds = make_labels(feats, cfg.labels)
    r = cfg.risk
    last_min = (int(r.no_entry_after[:2]) - 9) * 60 + int(r.no_entry_after[3:]) - 30
    ds = ds[ds["label_ok"] & (ds["minute"] >= r.no_entry_first_minutes) & (ds["minute"] <= last_min)]
    return bars, ds.reset_index(drop=True)


def walk_forward_predict(ds: pd.DataFrame, cfg: Config, log=print):
    days = sorted(ds["date"].unique())
    folds = walk_forward(days, cfg.wf, cfg.model.inner_val_frac)
    oos, fold_info, last_models = [], [], None
    for f in folds:
        assert_no_overlap(f)
        tr, va, te = (ds[ds["date"].isin(x)] for x in (f.train_days, f.val_days, f.test_days))
        te = te.copy()
        te["fold"] = f.k
        info = {"fold": f.k, "train_start": str(f.train_days[0]), "train_end": str(f.val_days[-1]),
                "test_start": str(f.test_days[0]), "test_end": str(f.test_days[-1]),
                "n_train": int(len(tr) + len(va)), "n_test": int(len(te))}
        models = {}
        for side in ("long", "short"):
            y = f"y_{side}"
            m = ProbModel(cfg.model).fit(tr[FEATURES], tr[y].to_numpy(), va[FEATURES], va[y].to_numpy())
            p_val, p_te = m.predict_proba(va[FEATURES]), m.predict_proba(te[FEATURES])
            te[f"p_{side}"] = p_te
            te[f"pq_{side}"] = percentile_rank(p_val, p_te)   # causal: reference = train-period
            yt = te[y].to_numpy()
            base = np.full_like(p_te, tr[y].mean())
            lr = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
            lr.fit(pd.concat([tr, va])[FEATURES], pd.concat([tr, va])[y])
            p_lr = lr.predict_proba(te[FEATURES])[:, 1]
            info[side] = {
                "base_rate_train": float(tr[y].mean()), "base_rate_test": float(yt.mean()),
                "auc": float(roc_auc_score(yt, p_te)) if 0 < yt.mean() < 1 else None,
                "auc_logit": float(roc_auc_score(yt, p_lr)) if 0 < yt.mean() < 1 else None,
                "logloss": float(log_loss(yt, np.clip(p_te, 1e-6, 1 - 1e-6), labels=[0, 1])),
                "logloss_base": float(log_loss(yt, np.clip(base, 1e-6, 1 - 1e-6), labels=[0, 1])),
                "brier": float(brier_score_loss(yt, p_te)), "brier_base": float(brier_score_loss(yt, base)),
            }
            models[side] = (m, va)
        hit = (te["y_long"] == 1) | (te["y_short"] == 1)
        info["dir_auc"] = float(roc_auc_score(te.loc[hit, "y_long"], te.loc[hit, "p_long"] - te.loc[hit, "p_short"]))
        fold_info.append(info)
        oos.append(te)
        last_models = models
        log(f"  fold {f.k}: test {info['test_start']}..{info['test_end']}  "
            f"AUC long={info['long']['auc']:.3f} short={info['short']['auc']:.3f}")
    return pd.concat(oos, ignore_index=True), fold_info, last_models


def calibration_table(oos: pd.DataFrame, side: str, bins: int = 10) -> list[dict]:
    q = pd.qcut(oos[f"p_{side}"], bins, duplicates="drop")
    g = oos.groupby(q, observed=True).agg(p_mean=(f"p_{side}", "mean"), hit=(f"y_{side}", "mean"),
                                          n=(f"y_{side}", "size"), gross=(f"gross_{side}", "mean"))
    return [{"p": float(r.p_mean), "hit": float(r.hit), "n": int(r.n), "gross_bps": float(r.gross * 1e4)}
            for r in g.itertuples()]


def _bt(oos, cfg, quantile, days):
    sig = select_signals(oos, quantile, cfg)
    return run_backtest(sig, cfg, days)


def robustness(oos: pd.DataFrame, cfg: Config, days: list) -> dict:
    out = {}
    q = cfg.wf.signal_quantile
    for mult in (0.0, 1.0, 2.0, 3.0):
        c = copy.deepcopy(cfg)
        c.costs.stress_multiplier = mult
        if mult == 0.0:
            c.costs.commission_per_share = c.costs.min_commission = 0.0
            c.costs.sec_fee_per_dollar_sold = c.costs.finra_taf_per_share = 0.0
        s = M.summary(_bt(oos, c, q, days), c.start_equity, cfg.n_configs_tried)
        out[f"cost_x{mult:g}"] = _brief(s)
    for qq in (0.80, 0.90, 0.95, 0.98, 0.99):
        out[f"quantile_{qq}"] = _brief(M.summary(_bt(oos, cfg, qq, days), cfg.start_equity, cfg.n_configs_tried))
    base = _bt(oos, cfg, q, days)
    tr = base["trades"]
    if len(tr):
        tr = tr.copy()
        tr["hour"] = pd.to_datetime(tr["entry_ts"]).dt.hour
        out["by_symbol"] = _group(tr, "symbol")
        out["by_fold"] = _group(tr, "fold")
        out["by_hour"] = _group(tr, "hour")
        out["by_direction"] = _group(tr.assign(side=np.where(tr.direction == 1, "long", "short")), "side")
        # Regimes defined from the QQQ session (known only after the fact -> descriptive only).
        qd = oos.groupby("date").agg(qqq_day=("qqq_ret_since_open", "last"), qvol=("qqq_vol_12", "mean"))
        tr = tr.join(qd, on="date")
        tr["trend"] = np.where(tr["qqq_day"] > 0, "QQQ up day", "QQQ down day")
        tr["vol_regime"] = np.where(tr["qvol"] > qd["qvol"].median(), "high-vol day", "low-vol day")
        out["by_trend"] = _group(tr, "trend")
        out["by_vol_regime"] = _group(tr, "vol_regime")
    return out


def _brief(s: dict) -> dict:
    keys = ("trades", "total_return", "sharpe", "max_drawdown", "profit_factor", "win_rate",
            "avg_net_bps", "avg_gross_bps", "p_value_trades")
    return {k: s.get(k) for k in keys}


def _group(tr: pd.DataFrame, col: str) -> list[dict]:
    g = tr.groupby(col).agg(trades=("pnl", "size"), pnl=("pnl", "sum"),
                            net_bps=("net_ret", lambda x: x.mean() * 1e4),
                            win=("pnl", lambda x: (x > 0).mean()))
    return [{"key": str(k), **{c: float(v) for c, v in r.items()}} for k, r in g.iterrows()]


def run(cfg: Config, refresh: bool = True, variants: bool = True, log=print) -> dict:
    t0 = time.time()
    log("Loading data + features + labels ...")
    bars, ds = build_dataset(cfg, refresh)
    log(f"  {len(ds):,} labelled rows, {ds['date'].nunique()} sessions, {ds['symbol'].nunique()} symbols")
    log("Walk-forward training ...")
    oos, folds, last_models = walk_forward_predict(ds, cfg, log)
    days = sorted(oos["date"].unique())
    base = _bt(oos, cfg, cfg.wf.signal_quantile, days)
    summ = M.summary(base, cfg.start_equity, cfg.n_configs_tried)

    c_frac = 4.5e-4
    rnd = random_entry_baseline(oos, max(summ.get("trades", 0), 30), 2000, c_frac)
    strat_mean = summ.get("avg_net_bps", 0) / 1e4 if summ.get("trades") else -1
    m, va = last_models["long"]
    imp = m.importance(va[FEATURES], va["y_long"].to_numpy())

    qqq = bars[(bars.symbol == "QQQ") & bars.date.isin(days)].groupby("date")["close"].last()
    qqq_bh = (qqq / qqq.iloc[0] * cfg.start_equity)

    variant = {}
    if variants:
        log("Variant: fixed +/-0.4% barrier (the original spec) ...")
        c2 = copy.deepcopy(cfg)
        c2.labels.barrier_mode = "fixed"
        _, ds2 = build_dataset(c2, refresh=False)
        oos2, folds2, _ = walk_forward_predict(ds2, c2, log=lambda *_: None)
        s2 = M.summary(_bt(oos2, c2, c2.wf.signal_quantile, sorted(oos2.date.unique())), c2.start_equity, c2.n_configs_tried)
        variant = {"fixed_0.4pct": _brief(s2),
                   "fixed_0.4pct_auc_long": float(np.nanmean([f["long"]["auc"] or np.nan for f in folds2])),
                   "fixed_0.4pct_base_rate": float(ds2["y_long"].mean())}

    hit = (oos["y_long"] == 1) | (oos["y_short"] == 1)
    edge = oos["p_long"] - oos["p_short"]
    dec = oos.assign(edge=edge).groupby(pd.qcut(edge, 10, labels=False), observed=True)
    diagnostics = {
        "corr_p_long_p_short": float(oos[["p_long", "p_short"]].corr().iloc[0, 1]),
        "auc_any_barrier": float(roc_auc_score(hit.astype(int), oos["p_long"])),
        "auc_direction": float(roc_auc_score(oos.loc[hit, "y_long"], edge[hit])),
        "edge_deciles": [{"decile": int(k) + 1, "edge": float(g["edge"].mean()),
                          "gross_long_bps": float(g["gross_long"].mean() * 1e4), "n": int(len(g))}
                         for k, g in dec],
    }
    report = {
        "diagnostics": diagnostics,
        "generated": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "runtime_s": round(time.time() - t0, 1), "backend": m.backend,
        "config": cfg.to_dict(), "data": data_quality_report(bars),
        "dataset": {"rows": int(len(ds)), "oos_rows": int(len(oos)), "oos_days": len(days),
                    "base_rate_long": float(ds["y_long"].mean()), "base_rate_short": float(ds["y_short"].mean()),
                    "median_barrier_bps": float(ds["barrier"].median() * 1e4)},
        "folds": folds, "summary": summ, "rejections": base["rejections"], "killed": base["killed"],
        "calibration": {"long": calibration_table(oos, "long"), "short": calibration_table(oos, "short")},
        "random_baseline": {"mean_bps": float(rnd.mean() * 1e4), "p95_bps": float(np.percentile(rnd, 95) * 1e4),
                            "p_value": float((rnd >= strat_mean).mean()),
                            "hist": np.histogram(rnd * 1e4, bins=40)[0].tolist(),
                            "edges": np.histogram(rnd * 1e4, bins=40)[1].round(2).tolist()},
        "importance": [{"feature": k, "value": float(v)} for k, v in imp.head(20).items()],
        "robustness": robustness(oos, cfg, days), "variants": variant,
        "equity": [{"date": str(d), "equity": float(e), "qqq": float(qqq_bh.get(d, np.nan))}
                   for d, e in base["equity"].items()],
    }
    return {"report": report, "oos": oos, "trades": base["trades"]}


def export_candidates(oos: pd.DataFrame, min_pq: float = 0.80) -> list[list]:
    """Compact OOS candidate rows for the browser simulator. Only causal fields."""
    keep = oos[(oos["pq_long"] >= min_pq) | (oos["pq_short"] >= min_pq)]
    t0 = pd.Timestamp("2000-01-01", tz="America/New_York")
    def mins(s):
        return ((pd.to_datetime(s) - t0).dt.total_seconds() // 60).astype(int)
    cols = pd.DataFrame({
        "e": mins(keep["entry_ts"]), "xl": mins(keep["exit_ts_long"]) + 5, "xs": mins(keep["exit_ts_short"]) + 5,
        "s": keep["symbol"], "px": keep["entry_px"].round(2), "b": (keep["barrier"] * 1e4).round(1),
        "pl": keep["p_long"].round(4), "ps": keep["p_short"].round(4),
        "ql": keep["pq_long"].round(4), "qs": keep["pq_short"].round(4),
        "gl": (keep["gross_long"] * 1e4).round(1), "gs": (keep["gross_short"] * 1e4).round(1),
        "rl": keep["reason_long"], "rs": keep["reason_short"],
    }).sort_values("e")
    return cols.values.tolist()
