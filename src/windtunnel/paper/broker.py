"""Brokers: where target weights become (simulated or real) orders.

`SimulatedBroker` uses `windtunnel.backtest.engine.plan_fill`, the same function the
backtest uses, so paper fills are sized and charged exactly like backtest fills.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from windtunnel.backtest.costs import BP, CostModel
from windtunnel.backtest.engine import Fill, plan_fill
from windtunnel.paper.store import Store


class Broker(Protocol):
    """The interface shared by the simulated, shadow and live brokers."""

    mode: str

    def account(self, price: float) -> tuple[float, float]:
        """Return (cash, units) of the capital this trader manages."""
        ...

    def rebalance(
        self, target_weight: float, price: float, bar_ts: pd.Timestamp, bar_vol: float
    ) -> Fill | None:
        """Trade towards ``target_weight`` at (approximately) ``price``."""
        ...

    def sync(self) -> None:
        """Poll or cancel outstanding orders and reconcile (no-op for simulation)."""
        ...


def client_order_id(strategy: str, bar_ts: pd.Timestamp, side: str) -> str:
    """Return a deterministic order id: the same bar and side always give the same id.

    Resubmitting after a crash is therefore detectable, and never double-trades.
    """
    return f"wt-{strategy[:12]}-{bar_ts.tz_convert('UTC'):%Y%m%d%H%M}-{side[0]}"


class SimulatedBroker:
    """Paper broker: fills instantly at the given price, with the backtest's cost model."""

    mode = "paper"

    def __init__(
        self,
        store: Store,
        costs: CostModel,
        *,
        strategy_name: str,
        initial_equity: float,
        rebalance_band: float,
    ) -> None:
        self.store = store
        self.costs = costs
        self.strategy_name = strategy_name
        self.rebalance_band = rebalance_band
        if store.get("cash") is None:
            store.set("cash", float(initial_equity))
            store.set("units", 0.0)
            store.event("start", f"new paper account with {initial_equity:g} quote currency")

    def account(self, price: float) -> tuple[float, float]:
        """Return the simulated (cash, units)."""
        return float(self.store.get("cash")), float(self.store.get("units"))

    def rebalance(
        self, target_weight: float, price: float, bar_ts: pd.Timestamp, bar_vol: float
    ) -> Fill | None:
        """Simulate the fill and persist the new account state."""
        cash, units = self.account(price)
        slip = self.costs.slippage_bps * BP + self.costs.vol_slippage_k * bar_vol
        fill = plan_fill(
            cash, units, price, target_weight, self.costs.fee_bps * BP, slip, self.rebalance_band
        )
        if fill is None:
            return None
        side = "buy" if fill.qty > 0 else "sell"
        cid = client_order_id(self.strategy_name, bar_ts, side)
        with self.store.transaction():
            self.store.record_order(
                client_id=cid,
                bar_ts=str(bar_ts),
                mode=self.mode,
                side=side,
                qty=abs(fill.qty),
                limit_price=None,
                status="filled",
            )
            self.store.record_fill(bar_ts=str(bar_ts), client_id=cid, **fill.as_trade_row())
            self.store.set("cash", fill.cash_after)
            self.store.set("units", fill.units_after)
        return fill

    def sync(self) -> None:
        """Nothing to do: simulated fills are instant."""
