"""Performance metrics. Stage 3 only has what walk-forward needs; Stage 5 adds the rest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """Return the annualised Sharpe ratio of simple returns, with a risk-free rate of 0.

    A series with zero variance (e.g. a strategy that never trades) scores 0.0, not NaN.
    That way it can be compared during parameter selection without special-casing.
    """
    r = returns.dropna().to_numpy()
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))
