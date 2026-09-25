"""Walk-forward evaluation: tune on the past, test on the unseen future, repeat.

::

    |---- train ----|-- test --|
               |---- train ----|-- test --|
                          |---- train ----|-- test --|

For each fold:

1. Every parameter set in ``strategy_cls.param_grid`` is backtested on the **train**
   window only, and scored with ``objective`` (Sharpe after costs by default).
2. The best set is run on the **test** window. Train-window bars are used only as
   indicator warm-up (no PnL is scored on them), so the first test-bar decision has
   the history it would have had in real time.
3. The test-window returns are stitched together into one out-of-sample equity curve.

Every (fold, params) evaluation is recorded in ``trials``. The number of *distinct*
configurations tried feeds the deflated Sharpe ratio in Stage 5.

Known conservative bias: each test fold starts flat, so a position carried across a
fold boundary is "re-bought" at the start of the next fold and pays one extra entry
cost per fold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from windtunnel.backtest.costs import CostModel
from windtunnel.backtest.engine import run_backtest
from windtunnel.backtest.sizing import Sizer
from windtunnel.evaluation.metrics import sharpe_ratio
from windtunnel.strategies.base import Strategy

Objective = Callable[[pd.Series, float], float]
"""Maps (train-window returns, periods_per_year) to a score. Higher is better."""


@dataclass
class Fold:
    """One train/test split and what was chosen in it."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_score: float
    test_returns: pd.Series


@dataclass
class WalkForwardResult:
    """The stitched out-of-sample result plus a full record of what was tried."""

    folds: list[Fold]
    oos_returns: pd.Series
    trials: pd.DataFrame
    periods_per_year: float

    @property
    def oos_equity(self) -> pd.Series:
        """Return the compounded out-of-sample equity curve, starting at 1.0."""
        return (1.0 + self.oos_returns).cumprod()

    @property
    def n_configs(self) -> int:
        """Return the number of distinct parameter sets tried (for multiple-testing corrections)."""
        return int(self.trials["params"].nunique())


def fold_indices(
    n: int, train_bars: int, test_bars: int, step_bars: int | None = None
) -> list[tuple[int, int, int]]:
    """Return ``(train_start, test_start, test_end)`` integer positions for rolling folds."""
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    step = step_bars or test_bars
    out = []
    s = 0
    while s + train_bars + test_bars <= n:
        out.append((s, s + train_bars, s + train_bars + test_bars))
        s += step
    if not out:
        raise ValueError(f"not enough bars ({n}) for one fold of {train_bars}+{test_bars}")
    return out


def _default_objective(returns: pd.Series, periods_per_year: float) -> float:
    return sharpe_ratio(returns, periods_per_year)


def walk_forward(
    bars: pd.DataFrame,
    strategy_cls: type[Strategy],
    sizer: Sizer,
    costs: CostModel,
    *,
    train_bars: int,
    test_bars: int,
    periods_per_year: float,
    step_bars: int | None = None,
    long_only: bool = True,
    objective: Objective = _default_objective,
) -> WalkForwardResult:
    """Run a rolling walk-forward evaluation. See the module docstring for the procedure."""
    grid = strategy_cls.grid() or [{}]
    folds: list[Fold] = []
    trial_rows: list[dict[str, Any]] = []
    for k, (tr0, te0, te1) in enumerate(fold_indices(len(bars), train_bars, test_bars, step_bars)):
        train = bars.iloc[tr0:te0]
        best_params: dict[str, Any] = grid[0]
        best_score = float("-inf")
        for params in grid:
            res = run_backtest(
                train, strategy_cls(**params), sizer, costs,
                periods_per_year=periods_per_year, long_only=long_only,
            )  # fmt: skip
            score = objective(res.returns, periods_per_year)
            trial_rows.append({"fold": k, "params": repr(sorted(params.items())), "score": score})
            if score > best_score:  # ties keep the earlier (first-listed) params
                best_score, best_params = score, params

        # test: warm up on the train bars, score only [te0, te1). Nothing after te1 exists here.
        window = bars.iloc[tr0:te1]
        test_res = run_backtest(
            window, strategy_cls(**best_params), sizer, costs,
            periods_per_year=periods_per_year, long_only=long_only, start=te0 - tr0,
        )  # fmt: skip
        folds.append(
            Fold(
                train_start=bars.index[tr0],
                train_end=bars.index[te0 - 1],
                test_start=bars.index[te0],
                test_end=bars.index[te1 - 1],
                best_params=best_params,
                train_score=best_score,
                test_returns=test_res.returns,
            )
        )

    oos = pd.concat([f.test_returns for f in folds])
    oos = oos[~oos.index.duplicated(keep="first")]  # overlapping tests if step < test_bars
    return WalkForwardResult(
        folds=folds,
        oos_returns=oos.rename("oos_returns"),
        trials=pd.DataFrame(trial_rows),
        periods_per_year=periods_per_year,
    )
