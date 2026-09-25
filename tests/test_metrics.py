import numpy as np
import pandas as pd
import pytest

from tests.helpers import FixedSignal, make_bars
from windtunnel.backtest.costs import ZERO_COSTS, CostModel
from windtunnel.backtest.engine import concat_results, run_backtest
from windtunnel.backtest.sizing import FixedFraction
from windtunnel.data.synthetic import gbm
from windtunnel.evaluation.metrics import (
    cagr,
    compute_metrics,
    drawdown,
    max_drawdown_duration,
    metrics_table,
    round_trip_returns,
    sharpe_ratio,
    sortino_ratio,
)


def _series(vals: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx)


def test_cagr_hand_computed() -> None:
    # doubling over exactly 2 "years" of 2 bars/year → 41.42% a year
    assert cagr(_series([0.0, 1.0, 0.0, 0.0]), periods_per_year=2) == pytest.approx(2**0.5 - 1)
    assert cagr(_series([-1.0, 0.1]), 365) == -1.0


def test_sharpe_and_sortino_hand_computed() -> None:
    r = _series([0.01, -0.01, 0.02, 0.0])
    mean, sd = np.mean(r), np.std(r, ddof=1)
    assert sharpe_ratio(r, 252) == pytest.approx(mean / sd * np.sqrt(252))
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert sortino_ratio(r, 252) == pytest.approx(mean / downside * np.sqrt(252))
    assert sharpe_ratio(_series([0.0, 0.0, 0.0]), 252) == 0.0


def test_drawdown_and_duration() -> None:
    r = _series([0.10, -0.50, 0.20, 1.00, -0.10])
    dd = drawdown(r)
    # equity 1.1, .55, .66, 1.32, 1.188 → peaks 1.1,1.1,1.1,1.32,1.32
    np.testing.assert_allclose(dd.to_numpy(), [0, -0.5, -0.4, 0, -0.1], atol=1e-12)
    # underwater from day 1 to day 3 (2 days), and from day 4 to the end (0 days, ongoing)
    assert max_drawdown_duration(r) == pd.Timedelta(days=2)


def test_initial_loss_counts_as_drawdown() -> None:
    assert drawdown(_series([-0.2, 0.1])).iloc[0] == pytest.approx(-0.2)


def test_round_trips_include_exit_bar() -> None:
    r = _series([0.0, 0.1, 0.1, -0.05, 0.0, 0.2, 0.0])
    w = _series([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    trips = round_trip_returns(r, w)
    assert len(trips) == 2
    assert trips.iloc[0] == pytest.approx(1.1 * 1.1 * 0.95 - 1)  # exit bar return included
    assert trips.iloc[1] == pytest.approx(1.2 * 1.0 - 1)  # still open at the end


def test_compute_metrics_on_known_backtest() -> None:
    bars = make_bars([100, 100, 110, 121, 121], [100, 110, 121, 121, 121])
    res = run_backtest(
        bars, FixedSignal([1, 1, 0, 0, 0]), FixedFraction(), ZERO_COSTS, periods_per_year=365
    )
    m = compute_metrics(res, "x")
    assert m.total_return == pytest.approx(121 / 100 - 1)
    assert m.n_trades == 2 and m.n_round_trips == 1 and m.win_rate == 1.0
    assert m.exposure == pytest.approx(2 / 5)
    assert m.cost_drag == pytest.approx(0.0)
    assert m.turnover > 0


def test_cost_drag_positive_with_costs() -> None:
    bars = gbm(400, seed=2)
    sig = FixedSignal([1.0 if (i // 20) % 2 else 0.0 for i in range(400)])
    m = compute_metrics(
        run_backtest(bars, sig, FixedFraction(), CostModel(), periods_per_year=365), "x"
    )
    m0 = compute_metrics(
        run_backtest(bars, sig, FixedFraction(), ZERO_COSTS, periods_per_year=365), "x"
    )
    assert m.cost_drag > 0
    assert m.cost_drag == pytest.approx(m0.cagr - m.cagr, rel=0.05)


def test_concat_results_compounds() -> None:
    bars = gbm(200, seed=4)
    a = run_backtest(
        bars.iloc[:100],
        FixedSignal([1.0] * 100),
        FixedFraction(),
        CostModel(),
        periods_per_year=365,
    )
    b = run_backtest(
        bars.iloc[100:],
        FixedSignal([1.0] * 100),
        FixedFraction(),
        CostModel(),
        periods_per_year=365,
    )
    both = concat_results([a, b])
    assert both.equity.iloc[-1] == pytest.approx(a.equity.iloc[-1] * b.equity.iloc[-1])
    assert both.n_trades == 2
    with pytest.raises(ValueError):
        concat_results([a, a])


def test_metrics_table_layout() -> None:
    bars = gbm(100, seed=1)
    res = run_backtest(
        bars, FixedSignal([1.0] * 100), FixedFraction(), CostModel(), periods_per_year=365
    )
    t = metrics_table([compute_metrics(res, "a"), compute_metrics(res, "b")])
    assert list(t.columns) == ["a", "b"] and "sharpe" in t.index
