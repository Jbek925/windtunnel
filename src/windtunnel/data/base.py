"""The pluggable `DataSource` interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from windtunnel.data.schema import Calendar


@runtime_checkable
class DataSource(Protocol):
    """Anything that can return canonical OHLCV bars for a symbol and timeframe.

    Implementations must:

    * return a frame that passes `windtunnel.data.schema.validate_schema`
      (UTC bar-open index, float64 OHLCV columns);
    * return **only completed bars**, never the bar that is still forming;
    * use public market data only. Data sources never take credentials.
    """

    name: str
    calendar: Calendar

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return completed bars whose open time is in ``[start, end)``.

        If ``end`` is ``None``, return every completed bar up to now.
        """
        ...

    def supported_timeframes(self) -> set[str]:
        """Return the timeframe strings this source can serve."""
        ...
