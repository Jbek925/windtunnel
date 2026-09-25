import pandas as pd
import pytest

from tests.conftest import FakeExchange
from windtunnel.data.base import DataSource
from windtunnel.data.ccxt_source import CcxtSource
from windtunnel.data.prediction_stub import PredictionMarketSource
from windtunnel.data.schema import validate_schema
from windtunnel.data.synthetic import gbm
from windtunnel.data.yfinance_source import YFinanceSource


def _hourly() -> pd.DataFrame:
    return gbm(2500, timeframe="1h", start="2024-01-01", seed=4)


def test_ccxt_paginates_and_dedupes_overlaps() -> None:
    bars = _hourly()
    ex = FakeExchange(bars, overlap=3)
    now = bars.index[-1] + pd.Timedelta(hours=5)  # all bars complete
    src = CcxtSource("fake", exchange=ex, page_limit=400, now=lambda: now)
    out = src.fetch("BTC/USDT", "1h", bars.index[0])
    validate_schema(out)
    assert len(ex.calls) > 5  # really paginated
    pd.testing.assert_frame_equal(out, bars, check_freq=False)


def test_ccxt_drops_incomplete_last_bar() -> None:
    bars = _hourly()
    ex = FakeExchange(bars)
    # "now" is 30 minutes into the final bar, so that bar is still forming
    now = bars.index[-1] + pd.Timedelta(minutes=30)
    src = CcxtSource("fake", exchange=ex, page_limit=1000, now=lambda: now)
    out = src.fetch("BTC/USDT", "1h", bars.index[0])
    assert out.index[-1] == bars.index[-2]


def test_ccxt_respects_start_and_end() -> None:
    bars = _hourly()
    src = CcxtSource(
        "fake", exchange=FakeExchange(bars), now=lambda: bars.index[-1] + pd.Timedelta(days=1)
    )
    out = src.fetch("BTC/USDT", "1h", bars.index[100], bars.index[200])
    assert out.index[0] == bars.index[100]
    assert out.index[-1] == bars.index[199]  # end is exclusive


def test_ccxt_is_a_datasource_and_needs_no_credentials() -> None:
    src = CcxtSource("fake", exchange=FakeExchange(_hourly()))
    assert isinstance(src, DataSource)
    assert src.supported_timeframes() == {"1h", "1d"}
    assert src.calendar == "24/7"


def _fake_yf_history(frame: pd.DataFrame):  # type: ignore[no-untyped-def]
    def history(symbol: str, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
        return frame

    return history


def _yf_frame() -> pd.DataFrame:
    """What yfinance returns: capitalised columns, NY-local midnight index, extra columns."""
    idx = pd.DatetimeIndex(pd.bdate_range("2024-03-04", periods=5), name="Date").tz_localize(
        "America/New_York"
    )
    return pd.DataFrame(
        {
            "Open": [10.0, 11, 12, 13, 14],
            "High": [11.0, 12, 13, 14, 15],
            "Low": [9.0, 10, 11, 12, 13],
            "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "Volume": [100.0, 100, 100, 100, 100],
            "Dividends": [0.0] * 5,
            "Stock Splits": [0.0] * 5,
        },
        index=idx,
    )


def test_yfinance_converts_to_utc_trading_dates() -> None:
    now = pd.Timestamp("2024-03-20 12:00", tz="UTC")
    src = YFinanceSource(history_fn=_fake_yf_history(_yf_frame()), now=lambda: now)
    out = src.fetch("SPY", "1d", pd.Timestamp("2024-03-01", tz="UTC"))
    validate_schema(out)
    assert out.index[0] == pd.Timestamp("2024-03-04", tz="UTC")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 5


@pytest.mark.parametrize(
    ("now_ny", "expected_last"), [("15:00", "2024-03-07"), ("16:30", "2024-03-08")]
)
def test_yfinance_drops_todays_bar_until_session_closes(now_ny: str, expected_last: str) -> None:
    now = pd.Timestamp(f"2024-03-08 {now_ny}", tz="America/New_York").tz_convert("UTC")
    src = YFinanceSource(history_fn=_fake_yf_history(_yf_frame()), now=lambda: now)
    out = src.fetch("SPY", "1d", pd.Timestamp("2024-03-01", tz="UTC"))
    assert out.index[-1] == pd.Timestamp(expected_last, tz="UTC")


def test_yfinance_rejects_intraday() -> None:
    with pytest.raises(ValueError):
        YFinanceSource(history_fn=_fake_yf_history(_yf_frame())).fetch(
            "SPY", "1h", pd.Timestamp("2024-01-01", tz="UTC")
        )


def test_prediction_stub_is_read_only_placeholder() -> None:
    src = PredictionMarketSource()
    assert isinstance(src, DataSource)
    assert src.supported_timeframes() == set()
    with pytest.raises(NotImplementedError):
        src.fetch("ANY", "1h", pd.Timestamp("2024-01-01", tz="UTC"))
