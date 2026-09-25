"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pytest

from windtunnel.data.synthetic import gbm


@pytest.fixture
def daily_bars() -> pd.DataFrame:
    """500 days of driftless 24/7 GBM bars."""
    return gbm(500, seed=1)


@pytest.fixture
def etf_bars() -> pd.DataFrame:
    """500 business days of exchange-calendar GBM bars."""
    return gbm(500, sigma_ann=0.2, calendar="exchange", seed=2)


class FakeExchange:
    """Mimics the public part of a ccxt exchange: paginated ``fetch_ohlcv``.

    Serves candles from ``bars``, at most ``limit`` per call, starting at ``since``.
    Every call is logged so tests can inspect pagination.
    """

    timeframes: ClassVar[dict[str, str]] = {"1h": "1h", "1d": "1d"}
    enableRateLimit = True  # noqa: N815  (ccxt's attribute name)

    def __init__(self, bars: pd.DataFrame, overlap: int = 0) -> None:
        self.bars = bars
        self.overlap = overlap  # re-send this many earlier candles per page (as real APIs may)
        self.calls: list[dict[str, Any]] = []

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
    ) -> list[list[float]]:
        self.calls.append({"symbol": symbol, "timeframe": timeframe, "since": since})
        ms = (self.bars.index.as_unit("ms").asi8).tolist()
        start_i = 0 if since is None else next((i for i, t in enumerate(ms) if t >= since), len(ms))
        start_i = max(0, start_i - self.overlap) if self.calls[1:] else start_i
        chunk = self.bars.iloc[start_i : start_i + (limit or 500)]
        return [
            [float(t), *row]
            for t, row in zip(
                chunk.index.as_unit("ms").asi8.tolist(), chunk.to_numpy().tolist(), strict=True
            )
        ]
