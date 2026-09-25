import pandas as pd
import pytest

from windtunnel.config import RiskConfig
from windtunnel.paper.feed import is_stale
from windtunnel.paper.risk import RiskManager
from windtunnel.paper.store import Store

T0 = pd.Timestamp("2024-03-01 00:00", tz="UTC")


@pytest.fixture
def store(tmp_path) -> Store:  # type: ignore[no-untyped-def]
    return Store(tmp_path / "t.sqlite")


def test_max_weight_clip(store: Store) -> None:
    rm = RiskManager(RiskConfig(max_weight=0.5), store)
    d = rm.evaluate(T0, 100.0, 10.0, 10.0, proposed=1.0)
    assert d.target == 0.5 and d.can_trade and "clipped" in d.reasons[0]


def test_daily_loss_flattens_until_next_utc_day(store: Store) -> None:
    rm = RiskManager(RiskConfig(max_daily_loss=0.05, max_drawdown=0.9), store)
    assert rm.evaluate(T0, 100.0, 10, 10, 1.0).target == 1.0  # day starts at 100
    d = rm.evaluate(T0 + pd.Timedelta(hours=5), 94.0, 10, 10, 1.0)  # -6% intraday
    assert d.target == 0.0 and "daily loss" in d.reasons[0]
    # later the same day: still paused, even after recovering
    assert rm.evaluate(T0 + pd.Timedelta(hours=9), 99.0, 10, 10, 1.0).target == 0.0
    # next UTC day: trading resumes
    assert rm.evaluate(T0 + pd.Timedelta(days=1), 99.0, 10, 10, 1.0).target == 1.0


def test_drawdown_kill_switch_latches_and_survives_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "k.sqlite"
    rm = RiskManager(RiskConfig(max_drawdown=0.2, max_daily_loss=0.99), Store(path))
    rm.evaluate(T0, 100.0, 10, 10, 1.0)
    rm.evaluate(T0 + pd.Timedelta(days=1), 120.0, 10, 10, 1.0)  # new peak 120
    d = rm.evaluate(T0 + pd.Timedelta(days=2), 95.0, 10, 10, 1.0)  # -20.8% from peak
    assert d.target == 0.0 and "KILL SWITCH" in d.reasons[0]
    # "restart": brand-new objects on the same database
    rm2 = RiskManager(RiskConfig(max_drawdown=0.2, max_daily_loss=0.99), Store(path))
    assert rm2.killed
    assert rm2.evaluate(T0 + pd.Timedelta(days=3), 200.0, 10, 10, 1.0).target == 0.0
    rm2.reset_kill_switch()
    assert rm2.evaluate(T0 + pd.Timedelta(days=4), 200.0, 10, 10, 1.0).target == 1.0


def test_price_jump_blocks_trading(store: Store) -> None:
    rm = RiskManager(RiskConfig(max_bar_move=0.25), store)
    d = rm.evaluate(T0, 100.0, close=14.0, prev_close=10.0, proposed=1.0)
    assert not d.can_trade and "possible bad data" in d.reasons[0]
    assert rm.evaluate(T0, 100.0, close=11.0, prev_close=10.0, proposed=1.0).can_trade


def test_stale_detection() -> None:
    last_open = pd.Timestamp("2024-03-01 10:00", tz="UTC")  # 1h bar completes 11:00
    assert not is_stale(last_open, pd.Timestamp("2024-03-01 11:59", tz="UTC"), "1h", 1.5)
    assert is_stale(last_open, pd.Timestamp("2024-03-01 12:31", tz="UTC"), "1h", 1.5)
