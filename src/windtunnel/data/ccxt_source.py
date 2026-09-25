"""Crypto OHLCV from public exchange endpoints, via ccxt.

This module only makes **unauthenticated** calls (``fetch_ohlcv`` and ``load_markets``).
It never takes an API key.
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
    timeframe_to_timedelta,
    validate_schema,
)

_MS = pd.Timedelta(milliseconds=1)


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class CcxtSource:
    """Fetch completed OHLCV bars from a ccxt exchange's public API.

    Args:
        exchange_id: ccxt exchange id, e.g. ``"binance"``, ``"kraken"`` or ``"okx"``.
            Some exchanges geoblock certain countries. Pick one that serves yours.
        exchange: an already-constructed exchange object. Tests use this to inject a fake.
            If ``None``, an unauthenticated ``ccxt.<exchange_id>`` is created.
        page_limit: the maximum number of candles requested per call.
        now: clock function. Injectable so tests can pin "now".
    """

    calendar: Calendar = "24/7"

    def __init__(
        self,
        exchange_id: str = "binance",
        *,
        exchange: Any | None = None,
        page_limit: int = 1000,
        now: Callable[[], pd.Timestamp] = _now_utc,
    ) -> None:
        if exchange is None:
            import ccxt  # local import so tests never need it

            exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self._ex = exchange
        self.name = f"ccxt_{exchange_id}"
        self.exchange_id = exchange_id
        self.page_limit = page_limit
        self._now = now

    def supported_timeframes(self) -> set[str]:
        """Return the timeframes the exchange advertises that windtunnel also understands."""
        advertised = set(getattr(self._ex, "timeframes", None) or {})
        return {tf for tf in ("1m", "5m", "15m", "1h", "4h", "1d") if tf in advertised}

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Page through ``fetch_ohlcv``; return completed bars opening in ``[start, end)``.

        Overlapping pages are expected. Duplicate timestamps from overlaps are dropped,
        keeping the most recently fetched row. The currently forming bar (open time +
        timeframe > now) is always dropped, because its "close" is not final.
        """
        td = timeframe_to_timedelta(timeframe)
        start = as_utc(start)
        now = self._now()
        last_complete_open = now - td  # a bar opening after this has not closed yet
        stop = last_complete_open if end is None else min(as_utc(end), last_complete_open)

        rows: list[list[float]] = []
        since_ms = int(start.timestamp() * 1000)
        while True:
            page = self._ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=self.page_limit)
            if not page:
                break
            rows.extend(page)
            last_ms = int(page[-1][0])
            if pd.Timestamp(last_ms, unit="ms", tz="UTC") >= stop or last_ms < since_ms:
                break
            next_since = last_ms + int(td / _MS)
            if next_since <= since_ms:  # exchange ignored `since`; avoid infinite loop
                break
            since_ms = next_since  # ccxt's enableRateLimit spaces out the calls

        if not rows:
            return empty_frame()
        df = pd.DataFrame(rows, columns=[INDEX_NAME, *OHLCV_COLUMNS])
        df[INDEX_NAME] = pd.to_datetime(df[INDEX_NAME], unit="ms", utc=True).dt.as_unit("ns")
        df = df.drop_duplicates(subset=INDEX_NAME, keep="last").set_index(INDEX_NAME)
        df = df.sort_index().astype("float64")
        mask = (df.index >= start) & (df.index <= last_complete_open)
        if end is not None:
            mask &= df.index < as_utc(end)
        df = df.loc[mask]
        validate_schema(df)
        return df
