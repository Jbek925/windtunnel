"""Daily ETF bars from Yahoo Finance, via yfinance (unofficial and public; no key needed).

Conventions and caveats:

* Prices are **adjusted** for splits and dividends (``auto_adjust=True``), so returns
  include dividends. Adjusted history is rewritten back in time after every dividend,
  so cached values can differ from a later re-download. That is expected.
* Each daily bar is labelled with 00:00 UTC of its trading date. The real session is
  roughly 14:30-21:00 UTC. The engine only relies on ordering (signal at close t, fill
  at open t+1), so the label is a naming choice and not a timing error.
* Today's bar is dropped until the US session has closed (after 16:15 New York time).
* yfinance is an unofficial scraper. It can break or return incomplete data, so always
  read the validation report.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from windtunnel.data.schema import (
    INDEX_NAME,
    OHLCV_COLUMNS,
    Calendar,
    as_utc,
    empty_frame,
    validate_schema,
)

HistoryFn = Callable[[str, pd.Timestamp, pd.Timestamp | None], pd.DataFrame]

_NY = "America/New_York"


def _yf_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    import yfinance as yf  # local import so tests never need network access

    kwargs: dict[str, Any] = {"start": start.strftime("%Y-%m-%d"), "interval": "1d"}
    if end is not None:
        kwargs["end"] = end.strftime("%Y-%m-%d")
    result: pd.DataFrame = yf.Ticker(symbol).history(auto_adjust=True, **kwargs)
    return result


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class YFinanceSource:
    """`DataSource` for daily ETF and equity bars from Yahoo Finance.

    Args:
        history_fn: a function returning yfinance-style history. Injectable for tests.
        now: clock function. Injectable for tests.
    """

    name = "yfinance"
    calendar: Calendar = "exchange"

    def __init__(
        self,
        history_fn: HistoryFn = _yf_history,
        now: Callable[[], pd.Timestamp] = _now_utc,
    ) -> None:
        self._history = history_fn
        self._now = now

    def supported_timeframes(self) -> set[str]:
        """Only daily bars are supported, because Yahoo's intraday history is short and patchy."""
        return {"1d"}

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return completed daily bars with trading date in ``[start, end)``."""
        if timeframe != "1d":
            raise ValueError("YFinanceSource only supports timeframe '1d'")
        start = as_utc(start)
        raw = self._history(symbol, start, end)
        if raw is None or raw.empty:
            return empty_frame()

        df = raw.rename(columns=str.lower).loc[:, list(OHLCV_COLUMNS)]
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is None:
            idx = idx.tz_localize(_NY)
        trading_dates = idx.tz_convert(_NY).normalize().tz_localize(None)
        df.index = (
            pd.DatetimeIndex(trading_dates).tz_localize("UTC").as_unit("ns").rename(INDEX_NAME)
        )
        df = df.astype("float64").sort_index()
        df = df.loc[~df.index.duplicated(keep="last")]

        now_ny = self._now().tz_convert(_NY)
        today = pd.Timestamp(now_ny.date()).tz_localize("UTC")
        session_closed = (now_ny.hour, now_ny.minute) >= (16, 15)
        last_allowed = today if session_closed else today - pd.Timedelta(days=1)
        mask = (df.index >= start.normalize()) & (df.index <= last_allowed)
        if end is not None:
            mask &= df.index < as_utc(end)
        df = df.loc[mask]
        validate_schema(df)
        return df
