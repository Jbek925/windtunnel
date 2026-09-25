import numpy as np
import pandas as pd

from windtunnel.data.validate import validate_bars


def test_clean_series_has_no_warnings(daily_bars: pd.DataFrame) -> None:
    rep = validate_bars(daily_bars, symbol="X", timeframe="1d", calendar="24/7")
    assert not rep.has_warnings, rep.summary()
    assert rep.n_bars == 500
    assert any("survivor" in n for n in rep.notes)


def test_detects_gap(daily_bars: pd.DataFrame) -> None:
    holed = daily_bars.drop(daily_bars.index[100:103])
    rep = validate_bars(holed, symbol="X", timeframe="1d", calendar="24/7")
    assert len(rep.gaps) == 1
    assert rep.missing_bars == 3
    assert rep.gaps[0].after == str(daily_bars.index[99])


def test_detects_duplicates(daily_bars: pd.DataFrame) -> None:
    dup = pd.concat([daily_bars, daily_bars.iloc[[10, 20]]])
    rep = validate_bars(dup, symbol="X", timeframe="1d", calendar="24/7")
    assert rep.duplicates == 2


def test_detects_outlier_but_does_not_modify_input(daily_bars: pd.DataFrame) -> None:
    spiky = daily_bars.copy()
    i = 300
    spiky.iloc[i, :4] = spiky.iloc[i, :4] * 3.0  # fat-finger print: close x3
    before = spiky.copy()
    rep = validate_bars(spiky, symbol="X", timeframe="1d", calendar="24/7")
    assert str(daily_bars.index[i]) in {o.timestamp for o in rep.outliers}
    pd.testing.assert_frame_equal(spiky, before)


def test_detects_ohlc_inconsistency_and_zero_volume(daily_bars: pd.DataFrame) -> None:
    bad = daily_bars.copy()
    bad.iloc[5, bad.columns.get_loc("high")] = bad["close"].iloc[5] * 0.5
    bad.iloc[6, bad.columns.get_loc("volume")] = 0.0
    rep = validate_bars(bad, symbol="X", timeframe="1d", calendar="24/7")
    assert rep.ohlc_inconsistent == 1
    assert rep.zero_volume == 1


def test_exchange_calendar_ignores_weekends_but_flags_long_gaps(etf_bars: pd.DataFrame) -> None:
    rep = validate_bars(etf_bars, symbol="SPY", timeframe="1d", calendar="exchange")
    assert rep.gaps == [] and rep.missing_business_days == 0

    holiday = etf_bars.drop(etf_bars.index[50])  # a single missing weekday, e.g. a holiday
    rep = validate_bars(holiday, symbol="SPY", timeframe="1d", calendar="exchange")
    assert rep.gaps == [] and rep.missing_business_days == 1

    outage = etf_bars.drop(etf_bars.index[100:110])  # two missing weeks
    rep = validate_bars(outage, symbol="SPY", timeframe="1d", calendar="exchange")
    assert len(rep.gaps) == 1 and rep.gaps[0].missing_bars == 10


def test_summary_and_dict_are_serialisable(daily_bars: pd.DataFrame) -> None:
    import json

    rep = validate_bars(daily_bars.iloc[:50], symbol="X", timeframe="1d", calendar="24/7")
    json.dumps(rep.to_dict())
    assert "X 1d" in rep.summary()


def test_tiny_input() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")], name="timestamp")
    one = pd.DataFrame({c: [1.0] for c in ["open", "high", "low", "close", "volume"]}, index=idx)
    rep = validate_bars(one, symbol="X", timeframe="1d", calendar="24/7")
    assert rep.n_bars == 1 and not np.isnan(rep.n_bars)
