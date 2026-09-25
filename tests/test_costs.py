import pandas as pd
import pytest

from tests.helpers import FixedSignal, make_bars
from windtunnel.backtest.costs import ZERO_COSTS, CostModel
from windtunnel.backtest.engine import run_backtest
from windtunnel.backtest.sizing import FixedFraction


def test_cost_rate_hand_computed() -> None:
    cm = CostModel(taker_fee_bps=10, maker_fee_bps=2, slippage_bps=5)
    assert cm.cost_rate() == pytest.approx(0.0015)  # 15 bps
    assert CostModel(
        taker_fee_bps=10, maker_fee_bps=2, slippage_bps=5, use_maker=True
    ).cost_rate() == pytest.approx(0.0007)
    # vol-scaled: 2% bar vol, k = 0.1 → +20 bps
    cm2 = CostModel(taker_fee_bps=10, slippage_bps=5, vol_slippage_k=0.1)
    assert cm2.cost_rate(0.02) == pytest.approx(0.0035)


def test_funding_per_bar() -> None:
    cm = CostModel(funding_rate_per_8h=0.0001)
    assert cm.funding_rate_per_bar(pd.Timedelta(days=1)) == pytest.approx(0.0003)
    assert cm.funding_rate_per_bar(pd.Timedelta(hours=1)) == pytest.approx(0.0001 / 8)


def test_scaled_doubles_trading_costs_not_funding() -> None:
    cm = CostModel(taker_fee_bps=10, slippage_bps=5, vol_slippage_k=0.1, funding_rate_per_8h=1e-4)
    d = cm.scaled(2)
    assert (d.taker_fee_bps, d.slippage_bps, d.vol_slippage_k) == (20, 10, 0.2)
    assert d.funding_rate_per_8h == cm.funding_rate_per_8h


def test_negative_costs_rejected() -> None:
    with pytest.raises(ValueError):
        CostModel(taker_fee_bps=-1)


def test_round_trip_cost_in_engine_matches_hand_calculation() -> None:
    # flat prices at 100 so all PnL is cost
    bars = make_bars([100.0] * 5, [100.0] * 5)
    cm = CostModel(taker_fee_bps=10, slippage_bps=5)
    # target 1 at close0 → buy at open1; target 0 at close2 → sell at open3
    res = run_backtest(
        bars, FixedSignal([1, 1, 0, 0, 0]), FixedFraction(), cm, periods_per_year=365
    )
    assert res.n_trades == 2
    # Buy sized so post-cost weight is exactly 100%: notional N = 1/(1+r), cost N·r.
    # Sell the same N at the same price, cost N·r again.
    r = 0.0015
    n = 1 / (1 + r)
    assert res.equity.iloc[-1] == pytest.approx((1 - r) / (1 + r), rel=1e-12)
    assert res.total_costs == pytest.approx(2 * n * r)
    assert res.costs["fees"].sum() == pytest.approx(2 * n * 0.001)
    assert res.costs["slippage"].sum() == pytest.approx(2 * n * 0.0005)


def test_full_buy_never_borrows_cash() -> None:
    bars = make_bars([100.0] * 3, [100.0] * 3)
    cm = CostModel(taker_fee_bps=50, slippage_bps=50)  # 1% total, exaggerated
    res = run_backtest(bars, FixedSignal([1, 1, 1]), FixedFraction(), cm, periods_per_year=365)
    trade = res.trades.iloc[0]
    cash_after = 1.0 - trade["notional"] - trade["fee"] - trade["slippage"]
    assert cash_after == pytest.approx(0.0, abs=1e-12)
    assert res.weights.iloc[1] == pytest.approx(1.0)


def test_funding_charged_on_held_notional() -> None:
    bars = make_bars([100.0] * 4, [100.0] * 4)
    cm = CostModel(taker_fee_bps=0, slippage_bps=0, funding_rate_per_8h=0.0001)
    res = run_backtest(bars, FixedSignal([1, 1, 1, 1]), FixedFraction(), cm, periods_per_year=365)
    # held 1.0 notional on bars 1,2,3 at 3 bps/day
    assert res.costs["funding"].sum() == pytest.approx(3 * 0.0003, rel=1e-6)


def test_vol_slippage_uses_only_past_volatility() -> None:
    cm = CostModel(vol_lookback=3)
    bars = make_bars([100, 101, 99, 100, 150, 100], [100, 101, 99, 100, 150, 100])
    vol = cm.trailing_bar_vol(bars)
    # the huge move into bar 4's close must not affect vol known at bar 4's open
    before = vol.iloc[4]
    bars2 = bars.copy()
    bars2.iloc[4, :4] = 100.0
    assert cm.trailing_bar_vol(bars2).iloc[4] == pytest.approx(before)


def test_zero_costs_model() -> None:
    assert ZERO_COSTS.cost_rate() == 0.0
