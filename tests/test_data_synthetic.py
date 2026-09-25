import numpy as np
import pandas as pd

from windtunnel.data.schema import validate_schema
from windtunnel.data.synthetic import gbm, shuffle_bars


def test_gbm_is_deterministic_and_valid() -> None:
    a, b = gbm(300, seed=7), gbm(300, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(gbm(300, seed=8))
    validate_schema(a)


def test_gbm_ohlc_consistent(daily_bars: pd.DataFrame) -> None:
    d = daily_bars
    assert (d["high"] >= d[["open", "close"]].max(axis=1)).all()
    assert (d["low"] <= d[["open", "close"]].min(axis=1)).all()


def test_gbm_has_gaps_between_close_and_next_open(daily_bars: pd.DataFrame) -> None:
    gap = np.log(daily_bars["open"].iloc[1:].to_numpy() / daily_bars["close"].iloc[:-1].to_numpy())
    assert np.abs(gap).mean() > 1e-4


def test_gbm_volatility_matches_parameter() -> None:
    bars = gbm(20_000, sigma_ann=0.5, seed=3)
    r = np.log(bars["close"]).diff().dropna()
    assert abs(r.std() * np.sqrt(365) - 0.5) < 0.02


def test_exchange_calendar_has_no_weekends(etf_bars: pd.DataFrame) -> None:
    assert (etf_bars.index.dayofweek < 5).all()


def test_shuffle_preserves_return_distribution(daily_bars: pd.DataFrame) -> None:
    shuffled = shuffle_bars(daily_bars, seed=5)
    validate_schema(shuffled)
    assert shuffled.index.equals(daily_bars.index)
    intra = np.sort(np.log(daily_bars["close"] / daily_bars["open"]).to_numpy())
    intra_s = np.sort(np.log(shuffled["close"] / shuffled["open"]).to_numpy())
    np.testing.assert_allclose(intra, intra_s, rtol=1e-9)
    assert not np.allclose(daily_bars["close"].to_numpy(), shuffled["close"].to_numpy())
