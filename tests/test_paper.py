"""Paper trader: must behave exactly like the backtest, and restart safely."""

from pathlib import Path

import pandas as pd
import pytest

from windtunnel.backtest.engine import run_backtest
from windtunnel.config import RiskConfig, TraderConfig, load_config
from windtunnel.data.synthetic import gbm
from windtunnel.paper.broker import SimulatedBroker
from windtunnel.paper.feed import ReplayFeed
from windtunnel.paper.runner import Runner, build_costs, build_strategy, compare_with_backtest
from windtunnel.paper.store import Store

LOOSE = RiskConfig(max_daily_loss=0.99, max_drawdown=0.99, max_bar_move=10.0)


def _cfg(tmp_path: Path, **kw) -> TraderConfig:  # type: ignore[no-untyped-def]
    base = dict(
        strategy="ma_trend",
        params={"lookback": 50, "fast": 1},
        history_bars=5000,
        db_path=str(tmp_path / "p.sqlite"),
        initial_equity=1000.0,
        risk=LOOSE,
    )
    base.update(kw)
    return TraderConfig(**base)


def _runner(cfg: TraderConfig, feed: ReplayFeed) -> Runner:
    store = Store(cfg.db_path)
    broker = SimulatedBroker(
        store,
        build_costs(cfg),
        strategy_name=cfg.strategy,
        initial_equity=cfg.initial_equity,
        rebalance_band=cfg.rebalance_band,
    )
    return Runner(cfg, feed, broker, store, alerter=lambda *a, **k: None, clock=feed.now)


def _drive(runner: Runner, feed: ReplayFeed, until: int) -> None:
    while feed.cursor < until:
        runner.step()
        feed.cursor += 1


def test_paper_equals_backtest_exactly(tmp_path: Path) -> None:
    bars = gbm(400, seed=12)
    cfg = _cfg(tmp_path)
    feed = ReplayFeed(bars, "1d", start=60)
    runner = _runner(cfg, feed)
    _drive(runner, feed, until=len(bars))

    strategy, sizer = build_strategy(cfg)
    # the runner first processes bar 59 (cursor 60 → completed bars [0, 60)): it decides at
    # close[59] and fills at open[60], i.e. a backtest scored from bar 60
    bt = run_backtest(
        bars,
        strategy,
        sizer,
        build_costs(cfg),
        periods_per_year=365,
        start=60,
        initial_equity=1000.0,
        rebalance_band=cfg.rebalance_band,
    )
    paper = runner.store.table("equity").set_index("bar_ts")["equity"]
    paper.index = pd.to_datetime(paper.index, utc=True).as_unit("ns")
    common = paper.index.intersection(bt.equity.index)
    assert len(common) > 300
    pd.testing.assert_series_equal(
        paper.loc[common], bt.equity.loc[common], check_names=False, check_freq=False, rtol=1e-10
    )
    assert len(runner.store.table("fills")) == bt.n_trades


def test_compare_with_backtest_reports_near_zero_gap(tmp_path: Path) -> None:
    bars = gbm(300, seed=13)
    cfg = _cfg(tmp_path)
    feed = ReplayFeed(bars, "1d", start=80)
    runner = _runner(cfg, feed)
    _drive(runner, feed, until=len(bars))
    both = compare_with_backtest(runner.store, bars, cfg)
    assert both["gap"].abs().max() < 1e-9


def test_restart_mid_run_matches_uninterrupted(tmp_path: Path) -> None:
    bars = gbm(300, seed=14)
    a_cfg = _cfg(tmp_path / "a")
    a_feed = ReplayFeed(bars, "1d", start=60)
    a = _runner(a_cfg, a_feed)
    _drive(a, a_feed, until=len(bars))

    b_cfg = _cfg(tmp_path / "b")
    b_feed = ReplayFeed(bars, "1d", start=60)
    b = _runner(b_cfg, b_feed)
    _drive(b, b_feed, until=170)
    b.store.close()
    b2 = _runner(b_cfg, b_feed)  # "crash" and restart on the same database
    assert b2.step().status in ("traded", "held")  # carries on from bar 169
    assert b2.step().status == "idle"  # same bar again: nothing happens, no double fill
    b_feed.cursor += 1
    _drive(b2, b_feed, until=len(bars))
    assert b2.store.get("cash") == pytest.approx(a.store.get("cash"), rel=1e-12)
    assert b2.store.get("units") == pytest.approx(a.store.get("units"), rel=1e-12)
    assert len(b2.store.table("fills")) == len(a.store.table("fills"))


def test_repeated_steps_are_idempotent(tmp_path: Path) -> None:
    bars = gbm(120, seed=15)
    cfg = _cfg(tmp_path)
    feed = ReplayFeed(bars, "1d", start=100)
    r = _runner(cfg, feed)
    assert r.step().status in ("traded", "held")
    for _ in range(5):
        assert r.step().status == "idle"
    assert len(r.store.table("equity")) == 1


def test_missed_bars_are_logged_and_only_latest_acted_on(tmp_path: Path) -> None:
    bars = gbm(200, seed=16)
    cfg = _cfg(tmp_path)
    feed = ReplayFeed(bars, "1d", start=100)
    alerts: list[str] = []
    store = Store(cfg.db_path)
    broker = SimulatedBroker(
        store,
        build_costs(cfg),
        strategy_name=cfg.strategy,
        initial_equity=1000.0,
        rebalance_band=0.02,
    )
    r = Runner(cfg, feed, broker, store, alerter=lambda k, m, *a: alerts.append(k), clock=feed.now)
    r.step()
    feed.cursor += 5  # "down" for 5 bars
    r.step()
    assert "missed_bars" in alerts
    assert len(store.table("equity")) == 2


def test_stale_feed_blocks_trading(tmp_path: Path) -> None:
    bars = gbm(150, seed=17)
    cfg = _cfg(tmp_path)
    feed = ReplayFeed(bars, "1d", start=100)
    alerts: list[str] = []
    store = Store(cfg.db_path)
    broker = SimulatedBroker(
        store,
        build_costs(cfg),
        strategy_name=cfg.strategy,
        initial_equity=1000.0,
        rebalance_band=0.02,
    )
    late = lambda: feed.now() + pd.Timedelta(days=3)  # noqa: E731
    r = Runner(cfg, feed, broker, store, alerter=lambda k, m, *a: alerts.append(k), clock=late)
    assert r.step().status == "stale"
    assert r.step().status == "stale"
    assert alerts.count("stale_data") == 1  # alerted once per stale bar, not every poll
    assert store.table("fills").empty


def test_kill_switch_flattens_in_runner(tmp_path: Path) -> None:
    # a crash: price falls 50% in steady steps; the trend rule is long before it
    import numpy as np

    from windtunnel.data.synthetic import _assemble

    n = 200
    up = np.linspace(np.log(100), np.log(200), 120)
    down = np.linspace(np.log(200), np.log(80), 80)
    log_close = np.concatenate([up, down])
    log_open = np.concatenate([[log_close[0]], log_close[:-1]])
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC", name="timestamp")
    bars = _assemble(log_open, log_close, np.random.default_rng(0), 0.001, idx.as_unit("ns"))
    cfg = _cfg(
        tmp_path,
        params={"lookback": 100, "fast": 1},
        risk=RiskConfig(max_drawdown=0.15, max_daily_loss=0.99, max_bar_move=10.0),
    )
    feed = ReplayFeed(bars, "1d", start=110)
    r = _runner(cfg, feed)
    _drive(r, feed, until=n)
    assert r.store.get("kill_switch") is True
    assert r.store.get("units") == 0.0
    assert r.store.table("fills")["qty"].iloc[-1] < 0  # the last trade was the forced exit


def test_example_config_loads() -> None:
    cfg = load_config(Path(__file__).parents[1] / "configs" / "paper_example.toml")
    assert cfg.mode == "paper" and cfg.live.max_live_notional == 50.0


def test_config_rejects_secrets_and_leverage(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text('mode = "live"\n[live]\napi_key = "abc"\n')
    with pytest.raises(ValueError, match="credential"):
        load_config(p)
    p.write_text("[risk]\nmax_weight = 2.0\n")
    with pytest.raises(ValueError, match="leverage"):
        load_config(p)
    p.write_text('mode = "yolo"\n')
    with pytest.raises(ValueError, match="mode"):
        load_config(p)


def test_cli_status_on_existing_db(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    from windtunnel.cli import main

    bars = gbm(150, seed=18)
    cfg = _cfg(tmp_path)
    feed = ReplayFeed(bars, "1d", start=100)
    _drive(_runner(cfg, feed), feed, until=110)
    conf = tmp_path / "c.toml"
    conf.write_text(f'db_path = "{cfg.db_path}"\n')
    assert main(["paper", "status", "--config", str(conf)]) == 0
    out = capsys.readouterr().out
    assert "last bar:" in out and "equity" in out
