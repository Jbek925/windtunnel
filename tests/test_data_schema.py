import numpy as np
import pandas as pd
import pytest

from windtunnel.data.schema import (
    SchemaError,
    as_utc,
    periods_per_year,
    timeframe_to_timedelta,
    to_canonical,
    validate_schema,
)


def test_synthetic_bars_pass_schema(daily_bars: pd.DataFrame) -> None:
    validate_schema(daily_bars)


def test_naive_index_rejected(daily_bars: pd.DataFrame) -> None:
    naive = daily_bars.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(SchemaError, match="tz"):
        validate_schema(naive)
    with pytest.raises(SchemaError, match="naive"):
        to_canonical(naive)


def test_non_utc_index_rejected(daily_bars: pd.DataFrame) -> None:
    ny = daily_bars.copy()
    ny.index = ny.index.tz_convert("America/New_York")
    with pytest.raises(SchemaError, match="UTC"):
        validate_schema(ny)
    validate_schema(to_canonical(ny))  # to_canonical converts to UTC explicitly


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda df: df.iloc[::-1], "sorted"),
        (lambda df: pd.concat([df, df.iloc[[5]]]).sort_index(), "duplicate"),
        (lambda df: df.assign(close=df["close"].where(df.index != df.index[3], np.nan)), "NaN"),
        (lambda df: df.assign(low=-df["low"]), "positive"),
        (lambda df: df.assign(volume=-df["volume"]), "volume"),
        (lambda df: df[["close", "open", "high", "low", "volume"]], "columns"),
    ],
)
def test_structural_violations(daily_bars: pd.DataFrame, mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SchemaError, match=match):
        validate_schema(mutate(daily_bars))


def test_to_canonical_lowercases_and_casts() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    raw = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [1, 1, 2],
            "Close": [2, 2, 3],
            "Volume": [0, 5, 6],
            "Dividends": [0, 0, 0],
        },
        index=idx,
    )
    out = to_canonical(raw)
    validate_schema(out)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_timeframes_and_annualisation() -> None:
    assert timeframe_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert periods_per_year("1d", "24/7") == 365
    assert periods_per_year("1h", "24/7") == 365 * 24
    assert periods_per_year("1d", "exchange") == 252
    with pytest.raises(ValueError):
        periods_per_year("1h", "exchange")
    with pytest.raises(ValueError):
        timeframe_to_timedelta("3w")


def test_as_utc_rejects_naive() -> None:
    with pytest.raises(ValueError, match="naive"):
        as_utc(pd.Timestamp("2024-01-01"))
    assert str(as_utc(pd.Timestamp("2024-01-01", tz="Europe/London")).tz) == "UTC"


def test_index_resolution_must_be_ns(daily_bars: pd.DataFrame) -> None:
    ms = daily_bars.copy()
    ms.index = ms.index.as_unit("ms")
    with pytest.raises(SchemaError, match="resolution"):
        validate_schema(ms)
    validate_schema(to_canonical(ms))
