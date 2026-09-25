import numpy as np
import pandas as pd
import pytest

from windtunnel.backtest.lookahead import check_causal
from windtunnel.backtest.sizing import FixedFraction, VolTarget
from windtunnel.data.synthetic import gbm


def test_fixed_fraction() -> None:
    bars = gbm(50, seed=0)
    sig = pd.Series(np.linspace(-1, 1, 50), index=bars.index)
    w = FixedFraction(0.5).target_weight(sig, bars, 365)
    assert w.iloc[-1] == pytest.approx(0.5) and w.iloc[0] == pytest.approx(-0.5)


def test_vol_target_scales_to_target() -> None:
    bars = gbm(3000, sigma_ann=0.6, seed=1)
    sig = pd.Series(1.0, index=bars.index)
    w = VolTarget(target_ann_vol=0.2, lookback=60).target_weight(sig, bars, 365)
    assert w.iloc[:59].isna().all()  # no weight until there is enough history
    assert w.iloc[100:].median() == pytest.approx(0.2 / 0.6, rel=0.1)


def test_vol_target_respects_max_leverage() -> None:
    calm = gbm(500, sigma_ann=0.05, seed=2)
    sig = pd.Series(1.0, index=calm.index)
    w = VolTarget(target_ann_vol=0.2, lookback=20, max_leverage=1.0).target_weight(sig, calm, 365)
    assert w.max() <= 1.0 + 1e-12


def test_vol_target_is_causal() -> None:
    bars = gbm(400, seed=4)
    vt = VolTarget(lookback=20)
    check_causal(
        lambda b: vt.target_weight(pd.Series(1.0, index=b.index), b, 365), bars, label="VolTarget"
    )
