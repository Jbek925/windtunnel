import json
from pathlib import Path

import pandas as pd

from tests.conftest import FakeExchange
from windtunnel.cli import main
from windtunnel.data.cache import ParquetCache
from windtunnel.data.ccxt_source import CcxtSource
from windtunnel.data.schema import validate_schema


def test_roundtrip_preserves_utc_and_dtypes(tmp_path: Path, daily_bars: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    cache.save(daily_bars, "src", "BTC/USDT", "1d", {"k": "v"})
    entry = cache.load("src", "BTC/USDT", "1d")
    assert entry is not None
    validate_schema(entry.bars)
    pd.testing.assert_frame_equal(entry.bars, daily_bars, check_freq=False)
    assert entry.meta == {"k": "v"}
    assert cache.path_for("src", "BTC/USDT", "1d").name == "1d.parquet"
    assert "BTC_USDT" in str(cache.path_for("src", "BTC/USDT", "1d"))


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert ParquetCache(tmp_path).load("src", "NOPE", "1d") is None


def test_incremental_update_fetches_only_new_bars(tmp_path: Path, daily_bars: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    start = daily_bars.index[0]
    first_half = daily_bars.iloc[:300]

    ex1 = FakeExchange(first_half)
    src1 = CcxtSource("fake", exchange=ex1, now=lambda: first_half.index[-1] + pd.Timedelta(days=1))
    bars1, rep1 = cache.update(src1, "BTC/USDT", "1d", start)
    assert len(bars1) == 300 and not rep1.has_warnings

    ex2 = FakeExchange(daily_bars)
    src2 = CcxtSource("fake", exchange=ex2, now=lambda: daily_bars.index[-1] + pd.Timedelta(days=1))
    bars2, rep2 = cache.update(src2, "BTC/USDT", "1d", start)
    # second update asked only from the last cached bar onward
    assert ex2.calls[0]["since"] == int(first_half.index[-1].timestamp() * 1000)
    pd.testing.assert_frame_equal(bars2, daily_bars, check_freq=False)
    assert not rep2.has_warnings
    assert not any("revised" in n for n in rep2.notes)

    meta = json.loads(
        cache.path_for("ccxt_fake", "BTC/USDT", "1d").with_suffix(".meta.json").read_text()
    )
    assert meta["rows"] == 500 and meta["index_convention"] == "bar open time, UTC"


def test_update_reports_revised_bars(tmp_path: Path, daily_bars: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    cache.save(daily_bars.iloc[:300], "ccxt_fake", "X", "1d", {})
    revised = daily_bars.copy()
    revised.iloc[299, :4] *= 1.01  # source changed the last cached bar
    src = CcxtSource(
        "fake", exchange=FakeExchange(revised), now=lambda: revised.index[-1] + pd.Timedelta(days=1)
    )
    bars, rep = cache.update(src, "X", "1d", daily_bars.index[0])
    assert any("1 previously cached bar" in n for n in rep.notes)
    assert bars.iloc[299]["close"] == revised.iloc[299]["close"]


def test_cli_validate_reads_cache(tmp_path: Path, daily_bars: pd.DataFrame, capsys) -> None:  # type: ignore[no-untyped-def]
    ParquetCache(tmp_path).save(daily_bars, "ccxt_binance", "BTC/USDT", "1d", {})
    rc = main(["--cache", str(tmp_path), "validate", "--source", "ccxt", "--symbol", "BTC/USDT"])
    assert rc == 0
    assert "500 bars" in capsys.readouterr().out
    assert (
        main(["--cache", str(tmp_path), "validate", "--source", "yfinance", "--symbol", "SPY"]) == 1
    )
