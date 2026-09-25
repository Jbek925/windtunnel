"""Performance metrics. Every figure is net of costs unless its name says otherwise.

Conventions: returns are simple per-bar returns, the risk-free rate is 0 (so Sharpe
here slightly flatters cash-like strategies when rates are high), and annualisation
uses ``periods_per_year`` (365 or 8760 for crypto, 252 for ETFs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from windtunnel.backtest.engine import BacktestResult

_EPS = 1e-12


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


def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """Return the annualised mean over annualised downside deviation (target 0)."""
    r = returns.dropna().to_numpy()
    if len(r) < 2:
        return 0.0
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if downside == 0:
        return 0.0
    return float(r.mean() / downside * np.sqrt(periods_per_year))


def cagr(returns: pd.Series, periods_per_year: float) -> float:
    """Return the compound annual growth rate implied by the returns."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    growth = float(np.prod(1.0 + r.to_numpy()))
    years = len(r) / periods_per_year
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def drawdown(returns: pd.Series) -> pd.Series:
    """Return the drawdown at each bar: equity / running peak - 1 (0 or negative)."""
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    peak = eq.cummax().clip(lower=1.0)  # the starting capital counts as a peak
    dd: pd.Series = eq / peak - 1.0
    return dd


def max_drawdown_duration(returns: pd.Series) -> pd.Timedelta:
    """Return the longest time spent below a previous equity peak (ongoing ones count too)."""
    dd = drawdown(returns)
    if len(dd) == 0:
        return pd.Timedelta(0)
    idx = pd.DatetimeIndex(dd.index)
    longest = pd.Timedelta(0)
    start: pd.Timestamp | None = None
    for ts, underwater in zip(idx, (dd < -_EPS).to_numpy(), strict=True):
        if underwater and start is None:
            start = ts
        elif not underwater and start is not None:
            longest = max(longest, ts - start)
            start = None
    if start is not None:
        longest = max(longest, idx[-1] - start)
    return longest


def round_trip_returns(returns: pd.Series, weights: pd.Series) -> pd.Series:
    """Return the compounded return of each round trip, from entry until back to flat.

    The bar on which the exit fills is included. Its return contains the close→open gap
    earned by the old position, and the exit costs. An open trade at the end counts too.
    """
    in_mkt = (weights.abs() > _EPS).to_numpy()
    r = returns.to_numpy()
    out: list[float] = []
    ends: list[pd.Timestamp] = []
    i, n = 0, len(r)
    while i < n:
        if not in_mkt[i]:
            i += 1
            continue
        j = i
        while j < n and in_mkt[j]:
            j += 1
        last = min(j, n - 1)  # include the exit bar
        out.append(float(np.prod(1.0 + r[i : last + 1]) - 1.0))
        ends.append(returns.index[last])
        i = j + 1
    return pd.Series(out, index=pd.DatetimeIndex(ends), name="round_trip_return", dtype=float)


@dataclass
class Metrics:
    """A standard set of performance figures, all net of costs."""

    label: str
    start: str
    end: str
    years: float
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_dd_days: float
    turnover: float
    """Annual turnover as a multiple of average equity (1.0 = account traded once a year)."""
    cost_drag: float
    """Gross CAGR minus net CAGR: what costs took, per year."""
    n_trades: int
    n_round_trips: int
    win_rate: float
    """Share of round trips that made money (NaN if there were none)."""
    exposure: float
    """Share of bars with a non-zero position."""

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a plain dict."""
        return asdict(self)


def compute_metrics(result: BacktestResult, label: str) -> Metrics:
    """Compute `Metrics` from a `BacktestResult`."""
    r = result.returns
    ppy = result.periods_per_year
    years = len(r) / ppy
    prev_eq = np.concatenate(
        [[result.config.get("initial_equity", 1.0)], result.equity.to_numpy()[:-1]]
    )
    per_bar_cost = result.costs.sum(axis=1).to_numpy()
    gross = pd.Series(r.to_numpy() + per_bar_cost / prev_eq, index=r.index)
    trips = round_trip_returns(r, result.weights)
    traded = float(result.trades["notional"].sum()) if len(result.trades) else 0.0
    mean_eq = float(result.equity.mean()) if len(result.equity) else 1.0
    return Metrics(
        label=label,
        start=str(r.index[0].date()) if len(r) else "",
        end=str(r.index[-1].date()) if len(r) else "",
        years=years,
        total_return=float(np.prod(1.0 + r.to_numpy()) - 1.0),
        cagr=cagr(r, ppy),
        ann_vol=float(r.to_numpy().std(ddof=1)) * float(np.sqrt(ppy)) if len(r) > 1 else 0.0,
        sharpe=sharpe_ratio(r, ppy),
        sortino=sortino_ratio(r, ppy),
        max_drawdown=float(drawdown(r).min()) if len(r) else 0.0,
        max_dd_days=max_drawdown_duration(r) / pd.Timedelta(days=1),
        turnover=traded / mean_eq / years if years > 0 else 0.0,
        cost_drag=cagr(gross, ppy) - cagr(r, ppy),
        n_trades=result.n_trades,
        n_round_trips=len(trips),
        win_rate=float((trips > 0).mean()) if len(trips) else float("nan"),
        exposure=float((result.weights.abs() > _EPS).mean()) if len(r) else 0.0,
    )


def metrics_table(metrics: list[Metrics]) -> pd.DataFrame:
    """Return several `Metrics` side by side (rows = metrics, columns = labels)."""
    return pd.DataFrame([m.to_dict() for m in metrics]).set_index("label").T
