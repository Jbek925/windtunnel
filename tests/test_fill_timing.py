"""The timing contract: decide at close t, fill at open t+1, gap accrues to the old position."""

import numpy as np
import pandas as pd
import pytest

from tests.helpers import FixedSignal, SameBarReturn, make_bars
from windtunnel.backtest.costs import ZERO_COSTS
from windtunnel.backtest.engine import run_backtest, simulate
from windtunnel.backtest.sizing import FixedFraction
from windtunnel.data.synthetic import gbm

OPENS = [100.0, 110.0, 120.0, 90.0, 95.0]
CLOSES = [105.0, 115.0, 100.0, 92.0, 97.0]


def _run(signal: list[float], **kw) -> object:  # type: ignore[no-untyped-def]
    return run_backtest(
        make_bars(OPENS, CLOSES), FixedSignal(signal), FixedFraction(), ZERO_COSTS,
        periods_per_year=365, **kw,
    )  # fmt: skip


def test_signal_at_t_only_affects_position_from_open_t_plus_1() -> None:
    res = _run([0, 1, 0, 0, 0])  # decide long at close of bar 1
    assert list(res.weights) == pytest.approx([0, 0, 1, 0, 0])
    assert res.trades["timestamp"].iloc[0] == res.equity.index[2]
    assert res.trades["price"].iloc[0] == OPENS[2]


def test_pnl_is_open_to_open_and_gap_goes_to_old_position() -> None:
    # long bought at open2 (120), sold at open3 (90): the close2→open3 gap (100→90) is ours
    res = _run([0, 1, 0, 0, 0])
    assert res.equity.iloc[-1] == pytest.approx(90 / 120)
    # equity at close2 marks at 100
    assert res.equity.iloc[2] == pytest.approx(100 / 120)


def test_first_bar_cannot_trade() -> None:
    res = _run([1, 1, 1, 1, 1])
    assert res.weights.iloc[0] == 0.0
    assert res.equity.iloc[0] == 1.0
    assert res.equity.iloc[-1] == pytest.approx(CLOSES[-1] / OPENS[1])


def test_long_only_clips_shorts() -> None:
    res = _run([-1, -1, -1, -1, -1])
    assert res.n_trades == 0
    # wide band: hold constant units (no rebalancing) so the PnL is easy to hand-check
    res_ls = _run([-1, -1, -1, -1, -1], long_only=False, rebalance_band=0.5)
    assert res_ls.weights.iloc[1] == pytest.approx(-1.0)
    # short from open1 (110) to close4 (97): gains 13/110 on unit notional
    assert res_ls.equity.iloc[-1] == pytest.approx(1 + (110 - 97) / 110)


def test_nan_signal_is_flat() -> None:
    res = _run([np.nan, np.nan, 1, 1, 1])
    assert list(res.weights) == pytest.approx([0, 0, 0, 1, 1])


def test_start_parameter_blocks_trading_before_start() -> None:
    bars = make_bars(OPENS, CLOSES)
    target = pd.Series([1.0] * 5, index=bars.index)
    res = simulate(bars, target, ZERO_COSTS, periods_per_year=365, start=2)
    assert res.equity.index[0] == bars.index[2]
    # decision at close1 (allowed: pre-test history) fills at open2
    assert res.weights.iloc[0] == pytest.approx(1.0)
    assert res.equity.iloc[-1] == pytest.approx(CLOSES[-1] / OPENS[2])


def test_same_bar_signal_has_no_edge_on_random_walk() -> None:
    """If the engine forgot the shift, trading on a bar's own return would look like
    a money machine. With the correct next-open fill it must be ~zero on a random walk."""
    bars = gbm(6000, sigma_ann=0.6, seed=11)
    res = run_backtest(bars, SameBarReturn(), FixedFraction(), ZERO_COSTS, periods_per_year=365,
                       long_only=False)  # fmt: skip
    r = res.returns.to_numpy()
    t_stat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    assert abs(t_stat) < 4

    # control: the same signal *without* the shift (the bug we guard against) is wildly profitable
    wrong = simulate(bars, SameBarReturn().signal(bars).shift(-1), ZERO_COSTS, periods_per_year=365,
                     long_only=False)  # fmt: skip
    rw = wrong.returns.dropna().to_numpy()
    assert rw.mean() / (rw.std(ddof=1) / np.sqrt(len(rw))) > 20


def test_weights_drift_and_rebalancing_is_charged() -> None:
    from windtunnel.backtest.costs import CostModel

    bars = gbm(200, seed=3)
    cm = CostModel(taker_fee_bps=10, slippage_bps=0)
    half = run_backtest(bars, FixedSignal([1.0] * 200), FixedFraction(0.5), cm,
                        periods_per_year=365, rebalance_band=0.0)  # fmt: skip
    assert half.n_trades > 100  # with no band: re-balancing back to 50% every bar
    banded = run_backtest(bars, FixedSignal([1.0] * 200), FixedFraction(0.5), cm,
                          periods_per_year=365, rebalance_band=0.05)  # fmt: skip
    assert banded.n_trades < half.n_trades
    full = run_backtest(
        bars, FixedSignal([1.0] * 200), FixedFraction(1.0), cm, periods_per_year=365
    )
    assert full.n_trades == 1  # 100% invested barely drifts (only by its own fee)


def test_exit_to_flat_ignores_band() -> None:
    # enter 30% at open1; the price crashes 100 → 20 so the holding drifts to ~8% of equity.
    # The exit decision (0.08 → 0) is inside the 0.25 band, but exits must always execute.
    bars = make_bars([100, 100, 20, 20, 20], [100, 100, 20, 20, 20])
    res = run_backtest(bars, FixedSignal([0.3, 0.3, 0, 0, 0]), FixedFraction(), ZERO_COSTS,
                       periods_per_year=365, rebalance_band=0.25)  # fmt: skip
    assert res.weights.iloc[1] == pytest.approx(0.3)
    assert 0 < res.weights.iloc[2] < 0.25  # drifted, not topped up (inside band)
    assert res.weights.iloc[3] == 0.0  # exit executed despite the band


def test_signal_index_mismatch_rejected() -> None:
    class Bad(FixedSignal):
        def signal(self, bars: pd.DataFrame) -> pd.Series:
            return super().signal(bars).iloc[1:]

    with pytest.raises(ValueError, match=r"bars\.index"):
        run_backtest(make_bars(OPENS, CLOSES), Bad([1] * 5), FixedFraction(), ZERO_COSTS,
                     periods_per_year=365)  # fmt: skip
