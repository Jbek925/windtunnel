"""Synthetic OHLCV generators for tests and sanity checks.

These series have **no exploitable structure by construction**. A strategy that shows a
reliable edge on them is almost certainly exploiting a bug, such as lookahead, a cost
error or a fill-timing error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from windtunnel.data.schema import (
    INDEX_NAME,
    OHLCV_COLUMNS,
    Calendar,
    periods_per_year,
    timeframe_to_timedelta,
    validate_schema,
)


def _index(n: int, timeframe: str, start: str, calendar: Calendar) -> pd.DatetimeIndex:
    if calendar == "exchange":
        return pd.bdate_range(start=start, periods=n, tz="UTC", name=INDEX_NAME).as_unit("ns")
    td = timeframe_to_timedelta(timeframe)
    return pd.date_range(start=start, periods=n, freq=td, tz="UTC", name=INDEX_NAME).as_unit("ns")


def gbm(
    n: int = 2000,
    *,
    mu_ann: float = 0.0,
    sigma_ann: float = 0.6,
    timeframe: str = "1d",
    calendar: Calendar = "24/7",
    start: str = "2018-01-01",
    start_price: float = 100.0,
    gap_fraction: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate geometric-Brownian-motion OHLCV bars.

    Each bar's log return is split into a *gap* (close[t-1] to open[t]) carrying
    ``gap_fraction`` of the variance, and an *intrabar* part (open[t] to close[t]).
    The gap lets tests check that gap returns accrue to the position held
    *before* the fill. High and low extend beyond open and close by a random excursion.

    Args:
        n: number of bars.
        mu_ann: annualised drift of log price. 0 gives a driftless random walk.
        sigma_ann: annualised volatility of log returns.
        timeframe: bar size, e.g. ``"1d"`` or ``"1h"``.
        calendar: ``"24/7"`` or ``"exchange"`` (business days only).
        start: first bar's open time (UTC).
        start_price: price at the first open.
        gap_fraction: share of return variance that falls in the close→open gap.
        seed: RNG seed. The same seed always gives the same data.
    """
    rng = np.random.default_rng(seed)
    ppy = periods_per_year(timeframe, calendar)
    mu = mu_ann / ppy
    sigma = sigma_ann / np.sqrt(ppy)
    gap = rng.normal(mu * gap_fraction, sigma * np.sqrt(gap_fraction), n)
    intra = rng.normal(mu * (1 - gap_fraction), sigma * np.sqrt(1 - gap_fraction), n)
    gap[0] = 0.0
    log_open = np.log(start_price) + np.cumsum(gap + np.concatenate([[0.0], intra[:-1]]))
    log_close = log_open + intra
    return _assemble(log_open, log_close, rng, sigma, _index(n, timeframe, start, calendar))


def _assemble(
    log_open: np.ndarray,
    log_close: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    index: pd.DatetimeIndex,
    hi_exc: np.ndarray | None = None,
    lo_exc: np.ndarray | None = None,
    volume: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(index)
    if hi_exc is None:
        hi_exc = np.abs(rng.normal(0, sigma * 0.5, n))
    if lo_exc is None:
        lo_exc = np.abs(rng.normal(0, sigma * 0.5, n))
    if volume is None:
        volume = rng.lognormal(10, 0.5, n)
    o, c = np.exp(log_open), np.exp(log_close)
    df = pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) * np.exp(hi_exc),
            "low": np.minimum(o, c) * np.exp(-lo_exc),
            "close": c,
            "volume": volume,
        },
        index=index,
    )[list(OHLCV_COLUMNS)].astype("float64")
    validate_schema(df)
    return df


def shuffle_bars(bars: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Destroy any time-series structure in ``bars`` while keeping its return distribution.

    Each bar is decomposed into (gap return, intrabar return, high excursion, low
    excursion, volume). Those tuples are permuted as whole rows, and a new price path
    is rebuilt on the original timestamps. Fat tails and volatility level survive.
    Autocorrelation, trends and volatility clustering do not. Any "edge" measured on
    the result is noise or a bug.
    """
    validate_schema(bars)
    o = np.log(bars["open"].to_numpy())
    h = np.log(bars["high"].to_numpy())
    lo = np.log(bars["low"].to_numpy())
    c = np.log(bars["close"].to_numpy())
    gap = np.concatenate([[0.0], o[1:] - c[:-1]])
    intra = c - o
    hi_exc = h - np.maximum(o, c)
    lo_exc = np.minimum(o, c) - lo
    vol = bars["volume"].to_numpy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bars))
    gap, intra, hi_exc, lo_exc, vol = gap[perm], intra[perm], hi_exc[perm], lo_exc[perm], vol[perm]
    gap[0] = 0.0
    log_open = o[0] + np.cumsum(gap + np.concatenate([[0.0], intra[:-1]]))
    log_close = log_open + intra
    idx = pd.DatetimeIndex(bars.index)
    return _assemble(log_open, log_close, rng, 0.0, idx, hi_exc, lo_exc, vol)
