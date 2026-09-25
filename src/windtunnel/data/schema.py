"""The canonical OHLCV schema used everywhere in windtunnel.

A bar frame is a ``pandas.DataFrame`` with:

* index ``timestamp``: a ``DatetimeIndex`` (``datetime64[ns, UTC]``), one entry per bar, giving the
  bar's **open** time (the ccxt convention). It is strictly increasing and unique.
* columns ``open, high, low, close, volume``: all ``float64``, no NaN, prices > 0 and
  volume >= 0.

Timing contract: a signal computed at bar ``t`` may use everything up to and including
``close[t]``. The earliest it can trade is ``open[t+1]``.

`validate_schema` enforces the *structural* rules and raises on violation, because the
engine cannot run on a malformed frame. Data-quality issues (gaps, outliers, OHLC
inconsistencies) are *reported* by `windtunnel.data.validate` instead. They are flagged,
never silently fixed.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
INDEX_NAME = "timestamp"

Calendar = Literal["24/7", "exchange"]
"""``24/7`` for crypto, ``exchange`` for instruments that trade business days only."""

_TIMEFRAMES: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


class SchemaError(ValueError):
    """Raised when a frame violates the canonical OHLCV schema."""


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convert a timeframe string such as ``"1h"`` or ``"1d"`` to a ``Timedelta``."""
    try:
        return _TIMEFRAMES[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; expected one of {sorted(_TIMEFRAMES)}"
        ) from exc


def periods_per_year(timeframe: str, calendar: Calendar) -> float:
    """Return the number of bars per year, used to annualise returns and volatility.

    Crypto trades 24/7, so it has 365 days a year. Exchange-traded instruments have
    about 252 trading days a year. Intraday bars on an exchange calendar are not
    supported, because their session length varies.
    """
    td = timeframe_to_timedelta(timeframe)
    if calendar == "24/7":
        return pd.Timedelta(days=365) / td
    if td != pd.Timedelta(days=1):
        raise ValueError("exchange-calendar data is only supported at 1d")
    return 252.0


def as_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Return ``ts`` in UTC. Naive timestamps are rejected rather than silently assumed UTC."""
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        raise ValueError(f"timestamp {ts!r} is timezone-naive; pass an explicit UTC time")
    return out.tz_convert("UTC")


def empty_frame() -> pd.DataFrame:
    """Return an empty frame that satisfies the schema."""
    idx = pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME).as_unit("ns")
    return pd.DataFrame({c: pd.Series([], dtype="float64") for c in OHLCV_COLUMNS}, index=idx)


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce column names, dtypes and index name to canonical form.

    This converts representation only (lower-case names, float64, UTC). It does **not**
    sort, de-duplicate or fill anything. Those are data decisions, and callers must
    make them explicitly. A timezone-naive index is rejected rather than assumed UTC.
    """
    out = df.rename(columns=str.lower)
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    out = out.loc[:, list(OHLCV_COLUMNS)].astype("float64")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise SchemaError("index must be a DatetimeIndex")
    if out.index.tz is None:
        raise SchemaError("index is timezone-naive; localise it explicitly before use")
    out.index = out.index.tz_convert("UTC").as_unit("ns").rename(INDEX_NAME)
    return out


def validate_schema(df: pd.DataFrame) -> None:
    """Raise `SchemaError` if ``df`` violates the structural rules of the schema."""
    if list(df.columns) != list(OHLCV_COLUMNS):
        raise SchemaError(f"columns must be exactly {OHLCV_COLUMNS}, got {list(df.columns)}")
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise SchemaError("index must be a DatetimeIndex")
    if idx.tz is None or str(idx.tz) != "UTC":
        raise SchemaError(f"index must be tz=UTC, got {idx.tz}")
    if idx.unit != "ns":
        raise SchemaError(f"index resolution must be ns, got {idx.unit}")
    if idx.name != INDEX_NAME:
        raise SchemaError(f"index must be named {INDEX_NAME!r}")
    if not idx.is_unique:
        raise SchemaError("index has duplicate timestamps")
    if not idx.is_monotonic_increasing:
        raise SchemaError("index is not sorted ascending")
    for col in OHLCV_COLUMNS:
        if df[col].dtype != "float64":
            raise SchemaError(f"column {col!r} must be float64, got {df[col].dtype}")
    if df.isna().to_numpy().any():
        raise SchemaError("frame contains NaN")
    if (df[list(PRICE_COLUMNS)] <= 0).to_numpy().any():
        raise SchemaError("prices must be strictly positive")
    if (df["volume"] < 0).to_numpy().any():
        raise SchemaError("volume must be non-negative")
