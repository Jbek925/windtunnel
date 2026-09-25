"""Small hand-built fixtures for exact arithmetic tests."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from windtunnel.strategies.base import Strategy


def make_bars(opens: list[float], closes: list[float], freq: str = "1D") -> pd.DataFrame:
    """Build a valid bar frame from explicit opens/closes (high/low envelope them)."""
    idx = pd.date_range("2024-01-01", periods=len(opens), freq=freq, tz="UTC", name="timestamp")
    df = pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) * 1.01 for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) * 0.99 for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": [1.0] * len(opens),
        },
        index=idx.as_unit("ns"),
    ).astype("float64")
    return df


class FixedSignal(Strategy):
    """Returns a pre-set signal series (ignores bars beyond aligning on the index)."""

    name = "fixed"
    param_grid: ClassVar[dict[str, list[Any]]] = {}

    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = values

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(self.values[: len(bars)], index=bars.index, dtype="float64")


class SameBarReturn(Strategy):
    """Legit: sign of this bar's own close-vs-open move, known at close t."""

    name = "same_bar"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        return (bars["close"] > bars["open"]).astype(float) * 2 - 1


class PeekNextClose(Strategy):
    """DELIBERATELY LEAKY: uses tomorrow's close. Must be caught by the checks."""

    name = "peek"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        return (bars["close"].shift(-1) > bars["close"]).astype(float) * 2 - 1


class SmaTrend(Strategy):
    """Honest toy trend rule with a small grid (the real ones come in Stage 4)."""

    name = "sma_toy"
    param_grid: ClassVar[dict[str, list[Any]]] = {"lookback": [10, 20, 50]}

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        sma = bars["close"].rolling(self.params["lookback"]).mean()
        return (bars["close"] > sma).astype(float).where(sma.notna())
