"""Live public price feeds for the paper/live runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from windtunnel.data.schema import (
    INDEX_NAME,
    OHLCV_COLUMNS,
    empty_frame,
    timeframe_to_timedelta,
    validate_schema,
)


@dataclass
class FeedSnapshot:
    """Completed bars, plus the open of the bar that has just started (where fills happen)."""

    completed: pd.DataFrame
    forming_ts: pd.Timestamp | None
    forming_open: float | None


class Feed(Protocol):
    """Anything that can give the runner a `FeedSnapshot`."""

    timeframe: str

    def snapshot(self, n: int) -> FeedSnapshot:
        """Return up to ``n`` completed bars and the currently forming bar's open."""
        ...


def is_stale(
    last_completed_open: pd.Timestamp, now: pd.Timestamp, timeframe: str, factor: float
) -> bool:
    """Return True if the newest completed bar is overdue by more than ``factor`` bar lengths.

    The bar opening at ``t`` completes at ``t + td``, and the next one at ``t + 2·td``. We
    call the feed stale once ``now`` passes ``t + td + factor·td`` with nothing newer.
    """
    td = timeframe_to_timedelta(timeframe)
    return bool(now > last_completed_open + td + factor * td)


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class CcxtFeed:
    """Public OHLCV via ccxt. It needs no keys, even when the runner is trading live."""

    def __init__(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        *,
        exchange: Any | None = None,
        now: Callable[[], pd.Timestamp] = _now_utc,
    ) -> None:
        if exchange is None:
            import ccxt

            exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self._ex = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self._now = now

    def snapshot(self, n: int) -> FeedSnapshot:
        """Fetch the latest ``n + 1`` candles and split off the one still forming."""
        rows = self._ex.fetch_ohlcv(self.symbol, self.timeframe, limit=n + 1)
        if not rows:
            return FeedSnapshot(empty_frame(), None, None)
        df = pd.DataFrame(rows, columns=[INDEX_NAME, *OHLCV_COLUMNS])
        df[INDEX_NAME] = pd.to_datetime(df[INDEX_NAME], unit="ms", utc=True).dt.as_unit("ns")
        df = df.drop_duplicates(INDEX_NAME, keep="last").set_index(INDEX_NAME).sort_index()
        df = df.astype("float64")
        td = timeframe_to_timedelta(self.timeframe)
        now = self._now()
        done = df.loc[df.index + td <= now]
        forming = df.loc[df.index + td > now]
        validate_schema(done)
        if len(forming):
            return FeedSnapshot(
                done, pd.Timestamp(forming.index[0]), float(forming["open"].iloc[0])
            )
        return FeedSnapshot(done, None, None)


class ReplayFeed:
    """Replays historical bars one at a time. Used by tests and dry runs.

    At cursor ``k``, bars ``[0, k)`` are completed and bar ``k`` has just opened.
    """

    def __init__(self, bars: pd.DataFrame, timeframe: str, start: int) -> None:
        self.bars = bars
        self.timeframe = timeframe
        self.cursor = start

    def snapshot(self, n: int) -> FeedSnapshot:
        """Return the bars up to the cursor and the open of the cursor bar."""
        done = self.bars.iloc[max(0, self.cursor - n) : self.cursor]
        if self.cursor < len(self.bars):
            nxt = self.bars.iloc[self.cursor]
            return FeedSnapshot(
                done, pd.Timestamp(self.bars.index[self.cursor]), float(nxt["open"])
            )
        return FeedSnapshot(done, None, None)

    def now(self) -> pd.Timestamp:
        """Return a clock time just after the cursor bar opened."""
        td = timeframe_to_timedelta(self.timeframe)
        return pd.Timestamp(self.bars.index[self.cursor - 1]) + td + pd.Timedelta(seconds=5)
