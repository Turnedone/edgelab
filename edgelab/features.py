"""Point-in-time features. Rule: a feature row stamped `ts` may only use bars whose
CLOSE is <= ts. The decision is made at the close of bar ts and the order is filled
at the OPEN of bar ts+1 (see labels.py). tests/test_no_lookahead.py enforces this."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SECTOR

BARS_PER_DAY = 78


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    c, v = g["close"], g["volume"]
    r = np.log(c).diff()
    g["r1"] = r
    for n in (3, 6, 12, 24):
        g[f"ret_{n}"] = np.log(c / c.shift(n))
    g["vol_12"] = r.rolling(12).std()
    g["vol_48"] = r.rolling(48).std()
    g["vol_ratio"] = g["vol_12"] / g["vol_48"]
    g["sigma_30m"] = g["vol_48"] * np.sqrt(6)          # used for barrier sizing too
    for n in (3, 6, 12):
        g[f"z_ret_{n}"] = g[f"ret_{n}"] / (g["vol_48"] * np.sqrt(n))
    g["range_bar"] = (g["high"] - g["low"]) / c / g["vol_48"]

    day = g["date"]
    tp = (g["high"] + g["low"] + c) / 3
    g["vwap"] = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    g["vwap_dist"] = np.log(c / g["vwap"]) / g["sigma_30m"]
    hi = g["high"].groupby(day).cummax()
    lo = g["low"].groupby(day).cummin()
    g["day_range_pos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    day_open = g["open"].groupby(day).transform("first")
    g["ret_since_open"] = np.log(c / day_open)

    # Prior-session stats (shifted by one whole session -> no leakage).
    daily = g.groupby("date").agg(d_close=("close", "last"), d_high=("high", "max"),
                                  d_low=("low", "min"), d_open=("open", "first"))
    daily["prev_close"] = daily["d_close"].shift(1)
    daily["prev_range"] = ((daily["d_high"] - daily["d_low"]) / daily["d_close"]).shift(1)
    daily["prev_ret"] = np.log(daily["d_close"] / daily["d_close"].shift(1)).shift(1)
    daily["ret_5d"] = np.log(daily["d_close"] / daily["d_close"].shift(5)).shift(1)
    g = g.join(daily[["prev_close", "prev_range", "prev_ret", "ret_5d"]], on="date")
    g["gap"] = np.log(day_open / g["prev_close"])

    # Opening range (first 30 minutes). Only defined once the range is complete.
    mins = g["minute"]
    in_or = mins < 30
    or_hi = g["high"].where(in_or).groupby(day).transform("max")
    or_lo = g["low"].where(in_or).groupby(day).transform("min")
    after = mins >= 30
    g["or_break_up"] = np.where(after, np.log(c / or_hi) / g["sigma_30m"], np.nan)
    g["or_break_dn"] = np.where(after, np.log(c / or_lo) / g["sigma_30m"], np.nan)

    # Relative volume vs the same time-of-day over the previous 10 sessions.
    hist = v.groupby(g["minute"]).transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    g["rel_vol"] = np.log(v / hist)
    cumv = v.groupby(day).cumsum()
    cum_hist = cumv.groupby(g["minute"]).transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    g["rel_vol_cum"] = np.log(cumv / cum_hist)

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    g["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return g


def build_features(bars: pd.DataFrame, tradable: list[str]) -> pd.DataFrame:
    bars = bars.copy()
    open_ts = bars["ts"].dt.normalize() + pd.Timedelta(hours=9, minutes=30)
    bars["minute"] = ((bars["ts"] - open_ts).dt.total_seconds() // 60).astype(int)
    feats = pd.concat([_per_symbol(g) for _, g in bars.groupby("symbol", sort=False)],
                      ignore_index=True)

    # Market + sector context, joined on timestamp.
    ctx_cols = ["ret_3", "ret_6", "ret_12", "vol_12", "ret_since_open", "vwap_dist"]
    def ctx(sym, prefix):
        x = feats.loc[feats["symbol"] == sym, ["ts"] + ctx_cols].set_index("ts")
        return x.add_prefix(prefix)
    mkt = ctx("QQQ", "qqq_").join(ctx("SPY", "spy_"), how="outer")
    out = feats[feats["symbol"].isin(tradable)].join(mkt, on="ts")
    sec = feats.loc[feats["symbol"].isin(set(SECTOR.values())), ["ts", "symbol", "ret_6", "ret_since_open"]]
    sec = sec.rename(columns={"symbol": "sector", "ret_6": "sec_ret_6", "ret_since_open": "sec_ret_open"})
    out["sector"] = out["symbol"].map(SECTOR)
    out = out.merge(sec, on=["ts", "sector"], how="left")

    out["rel_ret_6_mkt"] = out["ret_6"] - out["qqq_ret_6"]
    out["rel_ret_6_sec"] = out["ret_6"] - out["sec_ret_6"]
    out["rel_open_mkt"] = out["ret_since_open"] - out["qqq_ret_since_open"]
    # Cross-sectional ranks at each timestamp (only uses same-time info).
    for col in ("ret_6", "rel_vol", "vwap_dist"):
        out[f"xs_rank_{col}"] = out.groupby("ts")[col].rank(pct=True)

    out["tod_sin"] = np.sin(2 * np.pi * out["minute"] / 390)
    out["tod_cos"] = np.cos(2 * np.pi * out["minute"] / 390)
    out["dow"] = out["ts"].dt.dayofweek
    return out.sort_values(["ts", "symbol"]).reset_index(drop=True)


FEATURES = [
    "ret_3", "ret_6", "ret_12", "ret_24", "z_ret_3", "z_ret_6", "z_ret_12",
    "vol_12", "vol_48", "vol_ratio", "range_bar", "vwap_dist", "day_range_pos",
    "ret_since_open", "gap", "prev_ret", "prev_range", "ret_5d", "or_break_up", "or_break_dn",
    "rel_vol", "rel_vol_cum", "rsi_14",
    "qqq_ret_3", "qqq_ret_6", "qqq_ret_12", "qqq_vol_12", "qqq_ret_since_open", "qqq_vwap_dist",
    "spy_ret_6", "sec_ret_6", "sec_ret_open", "rel_ret_6_mkt", "rel_ret_6_sec", "rel_open_mkt",
    "xs_rank_ret_6", "xs_rank_rel_vol", "xs_rank_vwap_dist", "tod_sin", "tod_cos", "minute", "dow",
]
