"""Position sizing: turning a signal in [-1, 1] into a target portfolio weight.

A weight of 1.0 means 100% of equity is in the asset, and 0.5 means half. Negative
weights are shorts, which the engine clips to 0 when ``long_only`` is set. Sizers
must be **causal**: the weight at bar t may only use data up to close t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Sizer(Protocol):
    """Anything that converts signals into target weights."""

    def target_weight(
        self, signal: pd.Series, bars: pd.DataFrame, periods_per_year: float
    ) -> pd.Series:
        """Return the target weight at each bar, using data up to that bar's close."""
        ...


@dataclass(frozen=True)
class FixedFraction:
    """``weight = signal × fraction``. With ``fraction = 1``, a full signal means fully invested."""

    fraction: float = 1.0

    def target_weight(
        self, signal: pd.Series, bars: pd.DataFrame, periods_per_year: float
    ) -> pd.Series:
        """Scale the signal by a constant."""
        return signal * self.fraction


@dataclass(frozen=True)
class VolTarget:
    """Scale exposure so that the position's expected annualised volatility is ``target_ann_vol``.

    The weight is ``signal × min(target / realised_vol, max_leverage)``, where realised vol
    is the trailing std of close-to-close log returns over ``lookback`` bars, annualised.
    The cap stops the sizer levering up in suspiciously calm periods. For spot trading,
    keep ``max_leverage <= 1``.
    """

    target_ann_vol: float = 0.2
    lookback: int = 30
    max_leverage: float = 1.0

    def realised_vol(self, bars: pd.DataFrame, periods_per_year: float) -> pd.Series:
        """Return the annualised trailing volatility known at each bar's close."""
        logret = pd.Series(np.log(bars["close"].to_numpy()), index=bars.index).diff()
        daily = logret.rolling(self.lookback, min_periods=self.lookback).std()
        return daily * float(np.sqrt(periods_per_year))

    def target_weight(
        self, signal: pd.Series, bars: pd.DataFrame, periods_per_year: float
    ) -> pd.Series:
        """Return the vol-scaled weights. They are NaN (so flat) until enough history exists."""
        vol = self.realised_vol(bars, periods_per_year)
        scale = (self.target_ann_vol / vol).clip(upper=self.max_leverage)
        weights: pd.Series = signal * scale
        return weights
