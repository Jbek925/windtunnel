import numpy as np
import pandas as pd
import pytest

from windtunnel.backtest.costs import ZERO_COSTS, CostModel
from windtunnel.backtest.engine import run_backtest
from windtunnel.backtest.lookahead import check_no_lookahead
from windtunnel.backtest.sizing import FixedFraction, VolTarget
from windtunnel.backtest.walkforward import walk_forward
from windtunnel.data.synthetic import gbm
from windtunnel.strategies import (
    STRATEGIES,
    BuyAndHold,
    MovingAverageTrend,
    TimeSeriesMomentum,
    ZScoreMeanReversion,
    standard_benchmarks,
)
from windtunnel.strategies.mean_reversion import positions_from_z

ALL_CONFIGS = [(cls, p) for cls in STRATEGIES.values() for p in (cls.grid() or [{}])]


def _trend_bars(direction: float, n: int = 400) -> pd.DataFrame:
    bars = gbm(n, mu_ann=direction * 3.0, sigma_ann=0.05, seed=3)
    return bars


@pytest.mark.parametrize(("cls", "params"), ALL_CONFIGS, ids=lambda x: str(x))
def test_every_config_is_causal_bounded_and_aligned(cls, params) -> None:  # type: ignore[no-untyped-def]
    bars = gbm(700, seed=17)
    strat = cls(**params)
    check_no_lookahead(strat, bars, n_checks=25, min_history=5)
    sig = strat.signal(bars)
    assert sig.index.equals(bars.index)
    assert sig.dropna().between(-1, 1).all()


def test_parameter_budget() -> None:
    """Every extra parameter is an overfitting risk: at most 2 per strategy, small grids."""
    for cls in STRATEGIES.values():
        assert len(cls.param_grid) <= 2, cls
        assert len(cls.grid()) <= 12, cls


def test_ma_trend_follows_direction() -> None:
    up = MovingAverageTrend(lookback=50, fast=1).signal(_trend_bars(+1))
    down = MovingAverageTrend(lookback=50, fast=1).signal(_trend_bars(-1))
    assert up.iloc[:49].isna().all() and up.iloc[49:].notna().all()
    assert (up.iloc[100:] == 1).mean() > 0.95
    assert (down.iloc[100:] == -1).mean() > 0.95


def test_ma_trend_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        MovingAverageTrend(lookback=20, fast=20)
    with pytest.raises(TypeError):
        MovingAverageTrend(window=5)


def test_tsmom_hand_computed() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC", name="timestamp")
    closes = [10.0, 11.0, 9.0, 12.0, 12.0, 8.0]
    bars = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx
    )
    sig = TimeSeriesMomentum(lookback=2).signal(bars)
    # t=2: 9 vs 10 → -1; t=3: 12 vs 11 → +1; t=4: 12 vs 9 → +1; t=5: 8 vs 12 → -1
    assert sig.iloc[:2].isna().all()
    assert list(sig.iloc[2:]) == [-1, 1, 1, -1]


def test_mean_reversion_state_machine() -> None:
    z = np.array([np.nan, 0.0, -2.5, -1.0, -0.4, 0.3, 2.1, 1.0, 0.5, np.nan, -3.0])
    pos = positions_from_z(z, entry=2.0)
    expected = [np.nan, 0, 1, 1, 0, 0, -1, -1, 0, np.nan, 1]
    np.testing.assert_array_equal(pos, expected)


def test_mean_reversion_rejects_exit_beyond_entry() -> None:
    with pytest.raises(ValueError):
        positions_from_z(np.zeros(3), entry=0.4)


def test_mean_reversion_buys_dips_on_oscillating_series() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC", name="timestamp")
    idx = idx.as_unit("ns")
    close = 100 + 5 * np.sin(np.arange(300) / 5.0)
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                         "volume": 1.0}, index=idx)  # fmt: skip
    strat = ZScoreMeanReversion(lookback=20, entry=1.0)
    sig = strat.signal(bars)
    z = strat.zscore(bars)
    assert (sig[z < -1.2] == 1).all()  # deep below the mean → long
    assert (sig[z > 1.2] == -1).all()  # far above → short
    res = run_backtest(bars, strat, FixedFraction(), ZERO_COSTS, periods_per_year=365,
                       long_only=False)  # fmt: skip
    assert res.equity.iloc[-1] > 1.0  # a sine wave is the ideal case for this rule


def test_buy_and_hold_matches_price_return_minus_one_fee() -> None:
    bars = gbm(300, seed=5)
    cm = CostModel(taker_fee_bps=10, slippage_bps=0)
    res = run_backtest(bars, BuyAndHold(), FixedFraction(), cm, periods_per_year=365)
    assert res.n_trades == 1
    price_ret = bars["close"].iloc[-1] / bars["open"].iloc[1]
    assert res.equity.iloc[-1] == pytest.approx(price_ret / (1 + 0.001), rel=1e-9)


def test_standard_benchmarks() -> None:
    vt = VolTarget(target_ann_vol=0.15)
    b = standard_benchmarks(vt)
    assert [x.label for x in b] == ["buy & hold", "vol-target B&H (15%)"]
    assert b[1].sizer is vt


@pytest.mark.parametrize("cls", [MovingAverageTrend, TimeSeriesMomentum, ZScoreMeanReversion])
def test_no_edge_on_random_walk_walk_forward(cls) -> None:  # type: ignore[no-untyped-def]
    """Out-of-sample on a driftless random walk, no strategy should look significant."""
    bars = gbm(3000, sigma_ann=0.6, seed=99)
    wf = walk_forward(bars, cls, FixedFraction(), CostModel(), train_bars=730, test_bars=365,
                      periods_per_year=365, long_only=False)  # fmt: skip
    r = wf.oos_returns.to_numpy()
    t_stat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    assert t_stat < 3.0  # costs make negative the expected sign; a big positive = bug
