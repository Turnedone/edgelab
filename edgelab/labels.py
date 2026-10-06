"""Triple-barrier targets, defined exactly the way the strategy would trade them.

Decision at close of bar t -> market order filled at OPEN of bar t+1 (entry).
Long : take-profit limit at entry*(1+b), stop at entry*(1-b), time stop at close of
       bar t+H, never held past the session's last bar.
y_long = 1 iff the take-profit is hit before the stop and before the time stop.
Conservative conventions (they bias the backtest AGAINST us, on purpose):
  * TP and SL touched in the same 5m bar -> assume the STOP happened first.
  * A limit counts as filled only if price trades THROUGH it by `tp_touch_buffer_bps`.
  * A stop gapped through fills at the bar's open, not at the stop price.
The gross per-trade return (before costs) is stored so the backtest never re-derives
outcomes from data the model could see."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig


def _side(o, h, l, c, entry_idx, last_idx, b, H, buf, direction):
    n = len(o)
    y = np.zeros(n, dtype=np.int8)
    ret = np.full(n, np.nan)
    exit_k = np.full(n, -1)
    reason = np.zeros(n, dtype=np.int8)       # 1 tp, 2 sl, 3 time
    valid = entry_idx <= last_idx
    e = np.where(valid, o[np.minimum(entry_idx, n - 1)], np.nan)
    done = ~valid
    for k in range(H):
        j = entry_idx + k
        live = ~done & (j <= last_idx)
        jj = np.minimum(j, n - 1)
        if direction == 1:
            tp_px, sl_px = e * (1 + b), e * (1 - b)
            hit_sl = live & (l[jj] <= sl_px)
            hit_tp = live & (h[jj] >= tp_px * (1 + buf)) & ~hit_sl
            sl_fill = np.minimum(sl_px, o[jj]) if k > 0 else sl_px
            r_sl = sl_fill / e - 1
        else:
            tp_px, sl_px = e * (1 - b), e * (1 + b)
            hit_sl = live & (h[jj] >= sl_px)
            hit_tp = live & (l[jj] <= tp_px * (1 - buf)) & ~hit_sl
            sl_fill = np.maximum(sl_px, o[jj]) if k > 0 else sl_px
            r_sl = 1 - sl_fill / e
        ret = np.where(hit_tp, b, ret)
        ret = np.where(hit_sl, r_sl, ret)
        y[hit_tp] = 1
        reason[hit_tp], reason[hit_sl] = 1, 2
        exit_k[hit_tp | hit_sl] = jj[hit_tp | hit_sl]
        done |= hit_tp | hit_sl
        # Time stop: last bar of horizon or last bar of the session.
        t_exit = live & ~done & ((k == H - 1) | (j == last_idx))
        r_t = (c[jj] / e - 1) * direction
        ret = np.where(t_exit, r_t, ret)
        reason[t_exit] = 3
        exit_k[t_exit] = jj[t_exit]
        done |= t_exit
    return y, ret, exit_k, reason


def make_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    """df: features frame (one row per symbol x bar). Returns df with label columns."""
    out = []
    for sym, g in df.sort_values(["symbol", "ts"]).groupby("symbol", sort=False):
        g = g.copy()
        o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        pos = np.arange(len(g))
        last_of_day = g.groupby("date")["ts"].transform("size").to_numpy() - 1
        first_of_day = pos - g.groupby("date").cumcount().to_numpy()
        last_idx = first_of_day + last_of_day
        if cfg.barrier_mode == "fixed":
            b = np.full(len(g), cfg.fixed_barrier)
        else:
            b = np.clip(cfg.vol_mult * g["sigma_30m"].to_numpy(float), cfg.min_barrier, cfg.max_barrier)
        buf = cfg.tp_touch_buffer_bps / 1e4
        H = cfg.horizon_bars
        for name, d in (("long", 1), ("short", -1)):
            y, r, ek, rs = _side(o, h, l, c, pos + 1, last_idx, b, H, buf, d)
            g[f"y_{name}"], g[f"gross_{name}"], g[f"reason_{name}"] = y, r, rs
            g[f"exit_ts_{name}"] = g["ts"].to_numpy()[np.clip(ek, 0, len(g) - 1)]
        g["barrier"] = b
        g["entry_px"] = np.r_[o[1:], np.nan]
        g["entry_ts"] = np.r_[g["ts"].to_numpy()[1:], g["ts"].to_numpy()[-1:]]
        # Entry must be in the same session as the signal.
        g["label_ok"] = (pos + 1 <= last_idx) & np.isfinite(g["gross_long"]) & np.isfinite(b)
        out.append(g)
    return pd.concat(out).sort_values(["ts", "symbol"]).reset_index(drop=True)
