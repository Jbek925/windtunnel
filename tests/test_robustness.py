import numpy as np
import pandas as pd
import pytest

from tests.helpers import SmaTrend
from windtunnel.backtest.costs import CostModel
from windtunnel.backtest.sizing import FixedFraction
from windtunnel.data.synthetic import gbm
from windtunnel.evaluation.metrics import Metrics
from windtunnel.evaluation.robustness import (
    SharpeCI,
    bootstrap_sharpe_ci,
    bootstrap_sharpe_diff_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    parameter_sensitivity,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_indices,
    verdict,
)


def _returns(mean: float, sd: float, n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(rng.normal(mean, sd, n), index=idx)


def test_bootstrap_indices_shape_and_blocks() -> None:
    idx = stationary_bootstrap_indices(1000, 50, mean_block=10, rng=np.random.default_rng(0))
    assert idx.shape == (50, 1000) and idx.min() >= 0 and idx.max() < 1000
    # consecutive indices dominate (blocks), about 1 - 1/10 of steps
    steps = (np.diff(idx, axis=1) % 1000) == 1
    assert 0.85 < steps.mean() < 0.95


def test_bootstrap_ci_contains_true_sharpe() -> None:
    r = _returns(0.001, 0.01, 3000, seed=1)  # true annual Sharpe = 0.1 * sqrt(365) ≈ 1.91
    ci = bootstrap_sharpe_ci(r, 365, n_boot=500, seed=0)
    assert ci.low < 0.1 * np.sqrt(365) < ci.high
    assert ci.low < ci.sharpe < ci.high


def test_noise_ci_includes_zero() -> None:
    ci = bootstrap_sharpe_ci(_returns(0.0, 0.01, 1000, seed=2), 365, n_boot=500)
    assert not ci.excludes_zero


def test_psr_behaves() -> None:
    strong = _returns(0.002, 0.01, 2000, seed=3)
    noise = _returns(0.0, 0.01, 2000, seed=3)
    assert probabilistic_sharpe_ratio(strong) > 0.99
    assert 0.02 < probabilistic_sharpe_ratio(noise) < 0.98


def test_expected_max_sharpe_grows_with_trials() -> None:
    assert expected_max_sharpe(1, 0.05) == 0.0
    values = [expected_max_sharpe(n, 0.05) for n in (2, 10, 100, 1000)]
    assert values == sorted(values) and values[0] > 0


def test_dsr_penalises_more_trials() -> None:
    r = _returns(0.0008, 0.01, 1500, seed=4)
    few = deflated_sharpe_ratio(r, 365, n_trials=1)
    many = deflated_sharpe_ratio(r, 365, n_trials=200)
    assert few.dsr == pytest.approx(few.psr_vs_zero)
    assert many.dsr < few.dsr
    assert many.sr_threshold_annual > 0


def test_parameter_sensitivity_covers_grid() -> None:
    sens = parameter_sensitivity(
        gbm(500, seed=5), SmaTrend, FixedFraction(), CostModel(), periods_per_year=365
    )
    assert list(sens["lookback"]) == [10, 20, 50]
    assert {"sharpe", "n_trades"} <= set(sens.columns)


def _m(label: str, sharpe: float, cagr: float = 0.1) -> Metrics:
    return Metrics(
        label, "", "", 1.0, cagr, cagr, 0.2, sharpe, sharpe, -0.3, 10, 1.0, 0.01, 5, 2, 0.5, 0.5
    )


def test_verdict_says_so_plainly_when_benchmark_wins() -> None:
    lines = verdict(
        _m("s", 0.3),
        [_m("buy & hold", 0.8)],
        SharpeCI(0.3, -0.5, 1.1, 0.95),
        deflated_sharpe_ratio(_returns(0, 0.01, 500), 365, n_trials=6),
    )
    assert "did NOT beat buy-and-hold" in lines[0]
    assert any("includes zero" in ln for ln in lines)


def test_verdict_never_claims_edge_without_significance() -> None:
    lines = verdict(
        _m("s", 1.5),
        [_m("buy & hold", 0.5)],
        SharpeCI(1.5, -0.1, 3.0, 0.95),
        deflated_sharpe_ratio(_returns(0, 0.01, 500), 365, n_trials=6),
    )
    assert lines[0].startswith("Mixed result") and "no proven edge" in lines[0]


def test_diff_ci_identical_series_is_zero() -> None:
    r = _returns(0.001, 0.02, 800, seed=6)
    ci = bootstrap_sharpe_diff_ci(r, r, 365, n_boot=300)
    assert ci.sharpe == pytest.approx(0.0) and ci.low == pytest.approx(0.0)
    assert not ci.excludes_zero


def test_diff_ci_detects_a_clear_advantage_and_not_a_tiny_one() -> None:
    bench = _returns(0.0005, 0.03, 2000, seed=7)
    rng = np.random.default_rng(8)
    clearly_better = bench * 0.5 + 0.002 + rng.normal(0, 0.002, len(bench))
    slightly_better = bench + rng.normal(0.00002, 0.01, len(bench))  # ~0.7%/yr extra
    assert bootstrap_sharpe_diff_ci(clearly_better, bench, 365, n_boot=300).excludes_zero
    assert not bootstrap_sharpe_diff_ci(slightly_better, bench, 365, n_boot=300).excludes_zero


def test_verdict_does_not_claim_beating_holding_within_luck() -> None:
    """Regression: a strategy that made money and edged out buy & hold on one path must
    not get the top verdict unless it beat holding by more than luck."""
    strong_abs = SharpeCI(0.9, 0.14, 1.67, 0.95)
    dsr = deflated_sharpe_ratio(_returns(0.003, 0.01, 2000, seed=9), 365, n_trials=4)
    assert dsr.dsr > 0.95
    diff = SharpeCI(0.16, -0.45, 0.80, 0.95)
    lines = verdict(
        _m("s", 0.9), [_m("buy & hold", 0.74), _m("vt", 0.63)], strong_abs, dsr, diff_ci=diff
    )
    assert lines[0].startswith("Beat buy-and-hold on this historical path")
    assert "NOT proven better than simply holding" in lines[0]
    assert any("could be luck" in ln for ln in lines)
    top = verdict(
        _m("s", 0.9),
        [_m("buy & hold", 0.74)],
        strong_abs,
        dsr,
        diff_ci=SharpeCI(0.9, 0.3, 1.5, 0.95),
    )
    assert "beat buy-and-hold by more than luck" in top[0]
