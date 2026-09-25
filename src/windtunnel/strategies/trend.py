"""Trend-following strategies: bet that recent direction persists.

Economic story: slow-moving capital, herding and risk-premium dynamics have made trends
in many markets persist more often than a coin flip would suggest. The evidence is
strongest when diversified across dozens of futures markets, and weaker for any single
asset. Trend rules usually lose money in choppy, range-bound markets and win big in
sustained moves. Expect a low win rate with large winners.

All lookbacks are in **bars**, so 200 means 200 days on daily data but only about 8 days
on hourly data.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from windtunnel.strategies.base import Strategy


class MovingAverageTrend(Strategy):
    """Long when the fast moving average is above the slow one, short (or flat) when below.

    ``fast = 1`` means "price vs its ``lookback``-bar moving average", the classic
    200-day rule. The signal is +1 / -1, and NaN until the slow average exists.

    Params (2): ``lookback`` (slow SMA length) and ``fast`` (fast SMA length, < lookback).
    """

    name = "ma_trend"
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "lookback": [50, 100, 200],
        "fast": [1, 20],
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if not 1 <= self.params["fast"] < self.params["lookback"]:
            raise ValueError("need 1 <= fast < lookback")

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        """Return +1 if fast SMA > slow SMA, else -1. NaN during warm-up."""
        close = bars["close"]
        slow = close.rolling(self.params["lookback"]).mean()
        fast = close.rolling(self.params["fast"]).mean()
        sig = pd.Series(np.where(fast > slow, 1.0, -1.0), index=bars.index)
        return sig.where(slow.notna())


class TimeSeriesMomentum(Strategy):
    """Long if the asset rose over the last ``lookback`` bars, short (or flat) if it fell.

    This is the rule studied in Moskowitz, Ooi & Pedersen (2012), "Time series momentum".
    One parameter.
    """

    name = "ts_momentum"
    param_grid: ClassVar[dict[str, list[Any]]] = {"lookback": [20, 60, 120, 250]}

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        """Return the sign of the ``lookback``-bar return (+1, -1, or 0 if exactly flat)."""
        past = bars["close"].shift(self.params["lookback"])  # a *past* value, so this is causal
        ret = bars["close"] / past - 1.0
        return pd.Series(np.sign(ret), index=bars.index).where(past.notna())
