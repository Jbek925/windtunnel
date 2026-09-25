"""End-to-end honesty checks: the full pipeline must not find edges that aren't there."""

import pytest

from windtunnel.backtest.costs import CostModel
from windtunnel.data.synthetic import gbm, shuffle_bars
from windtunnel.evaluation.evaluate import evaluate
from windtunnel.strategies import MovingAverageTrend, TimeSeriesMomentum, ZScoreMeanReversion


@pytest.mark.parametrize("cls", [MovingAverageTrend, TimeSeriesMomentum, ZScoreMeanReversion])
@pytest.mark.parametrize("seed", [0, 1])
def test_pipeline_finds_no_proven_edge_on_random_walk(cls, seed: int) -> None:  # type: ignore[no-untyped-def]
    bars = gbm(1800, sigma_ann=0.6, seed=100 + seed)
    ev = evaluate(
        bars,
        cls,
        title="rw",
        periods_per_year=365,
        costs=CostModel(),
        train_bars=500,
        test_bars=250,
        n_boot=300,
        seed=seed,
    )
    assert ev.dsr.dsr < 0.95, ev.verdict  # never "survives multiple testing" on pure noise
    assert "survives the multiple-testing adjustment" not in ev.verdict[0]


def test_pipeline_on_shuffled_trending_data() -> None:
    """Shuffle a strongly trending series: the trend's *drift* survives (buy & hold still
    wins) but the time-structure is gone, so the trend rule should have no special edge."""
    bars = shuffle_bars(gbm(1800, mu_ann=0.5, sigma_ann=0.5, seed=7), seed=3)
    ev = evaluate(
        bars,
        MovingAverageTrend,
        title="shuffled",
        periods_per_year=365,
        costs=CostModel(),
        train_bars=500,
        test_bars=250,
        n_boot=300,
    )
    assert ev.dsr.dsr < 0.95


def test_pipeline_detects_a_real_edge() -> None:
    """Control: on data WITH persistent trends, the same pipeline must be able to say yes,
    otherwise the checks above prove nothing."""
    import numpy as np
    import pandas as pd

    from windtunnel.data.synthetic import _assemble

    rng = np.random.default_rng(0)
    n = 2500
    regime = np.repeat(rng.choice([-1.0, 1.0], size=n // 100 + 1), 100)[:n]  # 100-bar trends
    intra = regime * 0.006 + rng.normal(0, 0.01, n)
    log_open = np.log(100) + np.concatenate([[0.0], np.cumsum(intra[:-1])])
    idx = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC", name="timestamp")
    bars = _assemble(log_open, log_open + intra, rng, 0.01, idx.as_unit("ns"))
    ev = evaluate(
        bars,
        TimeSeriesMomentum,
        title="trending",
        periods_per_year=365,
        costs=CostModel(),
        train_bars=500,
        test_bars=250,
        n_boot=300,
        long_only=False,
    )
    assert ev.sharpe_ci.excludes_zero
    assert ev.strategy_metrics.sharpe > ev.benchmark_metrics[0].sharpe
