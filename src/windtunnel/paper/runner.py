"""The trading loop: one `Runner.step` per poll. It is idempotent and safe to restart.

On each **newly completed** bar ``t`` the runner does exactly what the backtest does:

1. Mark the account to market at ``close[t]`` and record an equity snapshot.
2. Compute the strategy's signal on history up to ``close[t]``, then the sizer's target.
3. Apply risk limits (kill switch, daily loss, bad-price guard, max weight).
4. Trade towards the target at the **open of bar t+1** (the bar that just started).
5. Mark bar ``t`` as processed, **in the same database transaction** as step 4's fill.

If the process dies at any point, the next start redoes the step for bar ``t`` or skips it,
and never double-fills. If several bars completed while it was down, it acts only on the
latest one (you can't trade in the past) and logs how many it skipped.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from windtunnel.backtest.costs import CostModel
from windtunnel.backtest.engine import run_backtest
from windtunnel.backtest.sizing import FixedFraction, Sizer, VolTarget
from windtunnel.config import TraderConfig
from windtunnel.data.schema import periods_per_year, timeframe_to_timedelta
from windtunnel.paper.alerts import Alerter
from windtunnel.paper.broker import Broker
from windtunnel.paper.feed import Feed, is_stale
from windtunnel.paper.risk import RiskManager
from windtunnel.paper.store import Store
from windtunnel.strategies import STRATEGIES
from windtunnel.strategies.base import Strategy

log = logging.getLogger("windtunnel.runner")


def build_costs(cfg: TraderConfig) -> CostModel:
    """Return the spot cost model described by the config (no funding: spot only)."""
    return CostModel(
        taker_fee_bps=cfg.fee_bps, maker_fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps
    )


def build_strategy(cfg: TraderConfig) -> tuple[Strategy, Sizer]:
    """Return the configured strategy instance and sizer."""
    strategy = STRATEGIES[cfg.strategy](**cfg.params)
    sizer: Sizer = (
        VolTarget(target_ann_vol=cfg.vol_target) if cfg.sizer == "vol" else FixedFraction(1.0)
    )
    return strategy, sizer


@dataclass
class StepResult:
    """What happened in one `Runner.step` (used by tests and the CLI's ``--once``)."""

    status: str
    bar_ts: pd.Timestamp | None = None
    target: float | None = None
    equity: float | None = None


class Runner:
    """Drives one strategy on one market. See the module docstring for the sequence."""

    def __init__(
        self,
        cfg: TraderConfig,
        feed: Feed,
        broker: Broker,
        store: Store,
        *,
        alerter: Callable[..., None] | None = None,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self.cfg = cfg
        self.feed = feed
        self.broker = broker
        self.store = store
        self.costs = build_costs(cfg)
        self.strategy, self.sizer = build_strategy(cfg)
        self.ppy = periods_per_year(cfg.timeframe, "24/7")
        self.risk = RiskManager(cfg.risk, store)
        self.alert = alerter or Alerter(store)
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.td = timeframe_to_timedelta(cfg.timeframe)
        store.event(
            "start",
            f"runner started in {broker.mode} mode: {self.strategy!r} on "
            f"{cfg.symbol} {cfg.timeframe}",
        )

    def step(self) -> StepResult:
        """Process the newest completed bar, if there is one we haven't handled yet."""
        self.broker.sync()
        snap = self.feed.snapshot(self.cfg.history_bars)
        bars = snap.completed
        if len(bars) < 2:
            return StepResult("no_data")
        last_ts = pd.Timestamp(bars.index[-1])
        now = self.clock()
        if is_stale(last_ts, now, self.cfg.timeframe, self.cfg.risk.stale_after_bars):
            if self.store.get("stale_alerted") != str(last_ts):
                self.alert(
                    "stale_data",
                    f"no new {self.cfg.timeframe} bar since {last_ts}; "
                    "not trading until data resumes",
                )
                self.store.set("stale_alerted", str(last_ts))
            return StepResult("stale", last_ts)
        done = self.store.get("last_bar")
        if done is not None and last_ts <= pd.Timestamp(done):
            return StepResult("idle", last_ts)
        if snap.forming_open is None or snap.forming_ts is None:
            return StepResult("waiting_for_next_open", last_ts)
        if done is not None:
            skipped = int(((bars.index > pd.Timestamp(done)) & (bars.index < last_ts)).sum())
            if skipped:
                self.alert(
                    "missed_bars",
                    f"{skipped} bar(s) completed while down; acting on the latest ({last_ts}) only",
                )

        close = float(bars["close"].iloc[-1])
        prev_close = float(bars["close"].iloc[-2])
        cash, units = self.broker.account(close)
        equity = cash + units * close
        weights = self.sizer.target_weight(self.strategy.signal(bars).clip(-1, 1), bars, self.ppy)
        proposed = float(weights.iloc[-1])
        proposed = 0.0 if not np.isfinite(proposed) else max(0.0, proposed)  # spot: long/flat

        with self.store.transaction():
            decision = self.risk.evaluate(last_ts + self.td, equity, close, prev_close, proposed)
            self.store.record_equity(
                bar_ts=str(last_ts),
                close=close,
                cash=cash,
                units=units,
                equity=equity,
                target_weight=decision.target,
            )
            if decision.can_trade:
                self.broker.rebalance(
                    decision.target,
                    snap.forming_open,
                    snap.forming_ts,
                    self.costs.next_bar_vol(bars),
                )
            self.store.set("last_bar", str(last_ts))
        for reason in decision.reasons:
            level = "CRITICAL" if "KILL" in reason else "WARNING"
            self.alert("risk", reason, level)
        return StepResult(
            "traded" if decision.can_trade else "held", last_ts, decision.target, equity
        )


def compare_with_backtest(store: Store, bars: pd.DataFrame, cfg: TraderConfig) -> pd.DataFrame:
    """Re-run the backtest over the paper period and line it up against the paper equity.

    Both are rebased to 1.0 at the first paper bar. In paper mode, with no risk limit
    triggered and enough history, the two should match to rounding. Any gap is a bug or
    a risk intervention, and the events table says which. In live mode, the gap measures
    real-world slippage, partial fills and missed orders.
    """
    eq = store.table("equity")
    if eq.empty:
        raise ValueError("no paper equity recorded yet")
    eq["bar_ts"] = pd.to_datetime(eq["bar_ts"], utc=True).dt.as_unit("ns")
    paper = eq.set_index("bar_ts")["equity"].sort_index()
    # The paper trader's first decision is at close[first] and fills at open[first + 1],
    # which is exactly a backtest whose scored period starts at first + 1.
    first = int(bars.index.searchsorted(paper.index[0]))
    strategy, sizer = build_strategy(cfg)
    bt = run_backtest(
        bars,
        strategy,
        sizer,
        build_costs(cfg),
        periods_per_year=periods_per_year(cfg.timeframe, "24/7"),
        start=first + 1,
        rebalance_band=cfg.rebalance_band,
    )
    both = pd.DataFrame({"paper": paper / paper.iloc[0]})
    bt_eq = bt.equity / float(bt.config["initial_equity"])
    both["backtest"] = bt_eq.reindex(both.index)
    both.loc[paper.index[0], "backtest"] = 1.0
    both["gap"] = both["paper"] - both["backtest"]
    return both


def setup_logging(path: str) -> None:
    """Log to a rotating file (5 × 5 MB) and to stderr."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (
        RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5),
        logging.StreamHandler(),
    ):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def run_forever(runner: Runner, poll_seconds: float, *, max_errors: int = 20) -> None:
    """Call ``runner.step()`` every ``poll_seconds`` until SIGTERM/SIGINT.

    Errors are logged and alerted, then retried on the next poll. After ``max_errors`` in
    a row it gives up, so a supervisor (systemd) can restart it or a human can look.
    """
    stop = threading.Event()

    def _stop(signum: int, _frame: Any) -> None:
        log.info("signal %s received; stopping after this step", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    errors = 0
    while not stop.is_set():
        try:
            res = runner.step()
            errors = 0
            if res.status in ("traded", "held"):
                log.info(
                    "bar %s: %s, target %.3f, equity %.2f",
                    res.bar_ts,
                    res.status,
                    res.target or 0.0,
                    res.equity or 0.0,
                )
        except Exception as exc:
            errors += 1
            runner.alert("error", f"{type(exc).__name__}: {exc} ({errors}/{max_errors})", "ERROR")
            if errors >= max_errors or type(exc).__name__ in (
                "ReconciliationError",
                "CapExceededError",
            ):
                raise
        stop.wait(poll_seconds)
    runner.store.event("stop", "runner stopped cleanly")
