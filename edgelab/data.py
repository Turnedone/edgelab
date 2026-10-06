"""Market data loading + caching + quality checks.

The free source (yfinance) only serves ~60 days of 5-minute bars, which is far too
short to establish an edge. It is used here so the pipeline runs end-to-end for
free. For real research swap `fetch_yfinance` for Polygon/Databento/Alpaca SIP
(see DESIGN.md section 2). Running this daily accumulates history in the cache.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
CACHE = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_yfinance(symbols: list[str], interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbols, period=period, interval=interval, auto_adjust=False,
                      progress=False, group_by="ticker", prepost=False, threads=True)
    frames = []
    for s in symbols:
        if s not in raw.columns.get_level_values(0):
            continue
        d = raw[s].dropna(how="all").copy()
        d.columns = [c.lower().replace(" ", "_") for c in d.columns]
        d = d.rename_axis("ts").reset_index()
        d["symbol"] = s
        frames.append(d[["ts", "symbol", "open", "high", "low", "close", "volume"]])
    return pd.concat(frames, ignore_index=True)


def load_bars(symbols: list[str], interval: str = "5m", period: str = "60d",
              refresh: bool = True) -> pd.DataFrame:
    """Long-format bars [ts, symbol, open, high, low, close, volume], ts in US/Eastern.
    New downloads are merged into the parquet cache so history grows over time."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"bars_{interval}.parquet"
    cached = pd.read_parquet(path) if path.exists() else None
    if refresh or cached is None:
        fresh = fetch_yfinance(symbols, interval, period)
        df = fresh if cached is None else pd.concat([cached, fresh], ignore_index=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
        df = df.drop_duplicates(["symbol", "ts"], keep="last")
        df.to_parquet(path, index=False)
    else:
        df = cached
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
    df = df[df["symbol"].isin(symbols)]
    return clean_bars(df)


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    t = df["ts"].dt.time
    df = df[(t >= dt.time(9, 30)) & (t < dt.time(16, 0))]
    # Drop a possibly-incomplete current session (bars still forming).
    now = pd.Timestamp.now(tz=ET)
    if now.time() < dt.time(16, 15):
        df = df[df["ts"].dt.date < now.date()]
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    bad |= (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    df = df[~bad & (df["volume"] > 0)]
    # Keep only sessions that are (nearly) complete for every symbol: half-days and
    # data holes are excluded rather than silently forward-filled.
    df["date"] = df["ts"].dt.date
    counts = df.groupby(["symbol", "date"]).size()
    full = counts[counts >= 70].reset_index()[["symbol", "date"]]
    df = df.merge(full, on=["symbol", "date"])
    return df.sort_values(["symbol", "ts"]).reset_index(drop=True)


def data_quality_report(df: pd.DataFrame) -> dict:
    per = df.groupby("symbol").agg(bars=("ts", "size"), days=("date", "nunique"),
                                   first=("ts", "min"), last=("ts", "max"))
    rets = df.groupby("symbol")["close"].pct_change().abs()
    return {
        "symbols": int(df["symbol"].nunique()),
        "sessions": int(df["date"].nunique()),
        "bars": int(len(df)),
        "first": str(df["ts"].min()),
        "last": str(df["ts"].max()),
        "max_abs_5m_return": float(np.nanmax(rets)),
        "per_symbol_days_min": int(per["days"].min()),
    }
