from typing import Any, ClassVar

import pandas as pd
import pytest

from tests.helpers import SmaTrend
from windtunnel.backtest.costs import CostModel
from windtunnel.backtest.sizing import FixedFraction
from windtunnel.backtest.walkforward import fold_indices, walk_forward
from windtunnel.data.synthetic import gbm


def test_fold_indices() -> None:
    assert fold_indices(100, 50, 20) == [(0, 50, 70), (20, 70, 90)]
    assert fold_indices(100, 50, 20, step_bars=10)[1] == (10, 60, 80)
    with pytest.raises(ValueError):
        fold_indices(10, 50, 20)


class SpyTrend(SmaTrend):
    """Records the last timestamp of every history it is shown."""

    seen: ClassVar[list[tuple[dict[str, Any], pd.Timestamp]]] = []

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        SpyTrend.seen.append((dict(self.params), bars.index[-1]))
        return super().signal(bars)


def test_parameter_selection_never_sees_test_data() -> None:
    bars = gbm(1000, seed=21)
    SpyTrend.seen.clear()
    scored_until: list[pd.Timestamp] = []

    def spy_objective(returns: pd.Series, ppy: float) -> float:
        scored_until.append(returns.index[-1])
        return float(returns.mean())

    res = walk_forward(
        bars,
        SpyTrend,
        FixedFraction(),
        CostModel(),
        train_bars=400,
        test_bars=200,
        periods_per_year=365,
        objective=spy_objective,
    )
    grid_size = len(SpyTrend.grid())
    assert len(res.folds) == 3
    assert len(scored_until) == 3 * grid_size
    for k, fold in enumerate(res.folds):
        # every objective call in fold k scored data strictly before its test window
        for ts in scored_until[k * grid_size : (k + 1) * grid_size]:
            assert ts < fold.test_start
        # in fold k, the grid_size train calls end before the test window; the single
        # test call (warm-up + test) ends exactly at test_end, using the chosen params
        calls = SpyTrend.seen[k * (grid_size + 1) : (k + 1) * (grid_size + 1)]
        assert all(last < fold.test_start for _, last in calls[:grid_size])
        assert calls[-1] == (fold.best_params, fold.test_end)


def test_oos_is_stitched_test_windows_only() -> None:
    bars = gbm(1000, seed=22)
    res = walk_forward(
        bars,
        SmaTrend,
        FixedFraction(),
        CostModel(),
        train_bars=400,
        test_bars=200,
        periods_per_year=365,
    )
    assert len(res.oos_returns) == 600
    assert res.oos_returns.index[0] == bars.index[400]
    assert res.oos_returns.index.is_unique and res.oos_returns.index.is_monotonic_increasing
    assert res.n_configs == 3
    assert len(res.trials) == 9
    assert res.oos_equity.iloc[-1] == pytest.approx((1 + res.oos_returns).prod())
