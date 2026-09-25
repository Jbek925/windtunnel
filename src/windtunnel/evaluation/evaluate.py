"""One call that runs the whole honest-evaluation pipeline for a strategy on one market."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from windtunnel.backtest.costs import CostModel
from windtunnel.backtest.engine import BacktestResult, run_backtest
from windtunnel.backtest.sizing import FixedFraction, Sizer, VolTarget
from windtunnel.backtest.walkforward import WalkForwardResult, walk_forward
from windtunnel.data.synthetic import shuffle_bars
from windtunnel.evaluation.metrics import Metrics, compute_metrics, sharpe_ratio
from windtunnel.evaluation.robustness import (
    DeflatedSharpe,
    SharpeCI,
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    parameter_sensitivity,
    verdict,
)
from windtunnel.strategies.base import Strategy
from windtunnel.strategies.benchmark import standard_benchmarks


@dataclass
class Evaluation:
    """Everything the report needs. Performance numbers are out-of-sample and net of costs."""

    title: str
    strategy_name: str
    wf: WalkForwardResult
    strategy_metrics: Metrics
    strategy_metrics_costs_x2: Metrics
    benchmark_results: dict[str, BacktestResult]
    benchmark_metrics: list[Metrics]
    sharpe_ci: SharpeCI
    dsr: DeflatedSharpe
    sensitivity: pd.DataFrame
    shuffled_sharpe: float
    verdict: list[str]
    notes: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def evaluate(
    bars: pd.DataFrame,
    strategy_cls: type[Strategy],
    *,
    title: str,
    periods_per_year: float,
    costs: CostModel,
    sizer: Sizer | None = None,
    train_bars: int,
    test_bars: int,
    long_only: bool = True,
    prior_trials: int = 0,
    n_boot: int = 2000,
    seed: int = 0,
    notes: list[str] | None = None,
) -> Evaluation:
    """Run walk-forward, benchmarks, robustness and sanity checks, and return an `Evaluation`.

    Args:
        bars: canonical OHLCV bars for one market.
        strategy_cls: the strategy class. Its ``param_grid`` is searched on each train window.
        title: a heading for the report, e.g. ``"BTC/USDT 1d"``.
        periods_per_year: bars per year, for annualisation.
        costs: cost model. It is also re-run at 2× for the stress test.
        sizer: position sizer (default: 100% of equity per unit of signal).
        train_bars, test_bars: walk-forward window lengths, in bars.
        long_only: clip shorts to flat (required for spot and live trading).
        prior_trials: other configurations you already tried on this data before
            this run (other strategies, symbols, timeframes). It is added to the
            multiple-testing count. Be honest here.
        n_boot: bootstrap replicates for the Sharpe CI.
        seed: RNG seed for the bootstrap and the shuffled-data sanity check.
        notes: extra caveats to print in the report (e.g. data validation warnings).
    """
    sizer = sizer or FixedFraction(1.0)
    wf_kwargs: dict[str, Any] = {
        "train_bars": train_bars,
        "test_bars": test_bars,
        "periods_per_year": periods_per_year,
        "long_only": long_only,
    }
    wf = walk_forward(bars, strategy_cls, sizer, costs, **wf_kwargs)
    wf2 = walk_forward(bars, strategy_cls, sizer, costs.scaled(2.0), **wf_kwargs)
    strat_m = compute_metrics(wf.oos_result, strategy_cls.name)
    strat_m2 = compute_metrics(wf2.oos_result, f"{strategy_cls.name} (costs ×2)")

    # Benchmarks over exactly the same out-of-sample window. Earlier bars are warm-up only.
    oos_start = int(bars.index.searchsorted(wf.oos_returns.index[0]))
    oos_end = int(bars.index.searchsorted(wf.oos_returns.index[-1]))
    vt = sizer if isinstance(sizer, VolTarget) else VolTarget()
    bench_results: dict[str, BacktestResult] = {}
    for b in standard_benchmarks(vt):
        bench_results[b.label] = run_backtest(
            bars.iloc[: oos_end + 1],
            b.strategy,
            b.sizer,
            costs,
            periods_per_year=periods_per_year,
            long_only=True,
            start=oos_start,
        )
    bench_m = [compute_metrics(r, label) for label, r in bench_results.items()]

    sens = parameter_sensitivity(
        bars, strategy_cls, sizer, costs, periods_per_year=periods_per_year, long_only=long_only
    )
    n_trials = wf.n_configs + prior_trials
    ci = bootstrap_sharpe_ci(wf.oos_returns, periods_per_year, n_boot=n_boot, seed=seed)
    dsr = deflated_sharpe_ratio(
        wf.oos_returns,
        periods_per_year,
        n_trials=n_trials,
        trial_sharpes_annual=sens["sharpe"].tolist(),
    )

    shuffled = walk_forward(shuffle_bars(bars, seed=seed), strategy_cls, sizer, costs, **wf_kwargs)
    shuffled_sharpe = sharpe_ratio(shuffled.oos_returns, periods_per_year)

    return Evaluation(
        title=title,
        strategy_name=strategy_cls.name,
        wf=wf,
        strategy_metrics=strat_m,
        strategy_metrics_costs_x2=strat_m2,
        benchmark_results=bench_results,
        benchmark_metrics=bench_m,
        sharpe_ci=ci,
        dsr=dsr,
        sensitivity=sens,
        shuffled_sharpe=shuffled_sharpe,
        verdict=verdict(strat_m, bench_m, ci, dsr, strat_m2),
        notes=list(notes or []),
        config={
            "strategy": strategy_cls.name,
            "param_grid": strategy_cls.param_grid,
            "sizer": repr(sizer),
            "costs": repr(costs),
            "train_bars": train_bars,
            "test_bars": test_bars,
            "long_only": long_only,
            "periods_per_year": periods_per_year,
            "n_trials": n_trials,
            "prior_trials": prior_trials,
            "bars": f"{bars.index[0].date()} → {bars.index[-1].date()} ({len(bars)} bars)",
        },
    )
