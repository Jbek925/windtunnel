"""The live (and shadow) spot broker for ccxt exchanges.

Read this file before running with real money. Guarantees:

* **Hard capital cap.** The trader only ever manages ``min(account value, max_live_notional)``,
  and a final check refuses any order that would push the position value above the cap.
* **Limit orders only.** Prices sit within ``limit_band_bps`` of the reference price, so a
  thin order book can't fill you at an absurd price. Unfilled orders are cancelled after
  ``order_timeout_minutes``. There is no market-order code path.
* **Spot, long/flat only.** Negative targets are clipped to 0, and it never sells more than
  it holds.
* **Idempotent.** Every order has a deterministic client id (strategy + bar + side). An id
  already in the store is never resubmitted.
* **Reconcile, don't repair.** At startup, if the real balance disagrees with what the
  store expects, or there are open orders this program didn't place, it raises
  `ReconciliationError` and the runner halts. It never trades to "fix" a mismatch.
* **Shadow mode** runs everything except ``create_order`` and ``cancel_order``, and logs
  what it would have sent.

Only these private exchange methods are ever called: ``fetch_balance``, ``create_order``,
``fetch_order``, ``cancel_order`` and ``fetch_open_orders``. A test enforces this.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from windtunnel.backtest.costs import BP, CostModel
from windtunnel.backtest.engine import Fill, plan_fill
from windtunnel.config import LiveConfig
from windtunnel.paper.broker import client_order_id
from windtunnel.paper.store import Store

log = logging.getLogger("windtunnel.live")

OUR_PREFIX = "wt-"


class ReconciliationError(RuntimeError):
    """The exchange state doesn't match what this program expects. A human must look."""


class CapExceededError(RuntimeError):
    """An order would have breached ``max_live_notional``. This should be unreachable."""


class LiveCcxtBroker:
    """Places real (``live``) or pretend (``shadow``) limit orders on a ccxt spot exchange."""

    def __init__(
        self,
        exchange: Any,
        symbol: str,
        store: Store,
        costs: CostModel,
        live_cfg: LiveConfig,
        *,
        strategy_name: str,
        rebalance_band: float,
        shadow: bool,
    ) -> None:
        self.ex = exchange
        self.symbol = symbol
        self.base, self.quote = symbol.split("/")
        self.store = store
        self.costs = costs
        self.cfg = live_cfg
        self.strategy_name = strategy_name
        self.rebalance_band = rebalance_band
        self.shadow = shadow
        self.mode = "shadow" if shadow else "live"
        self.ex.load_markets()
        self.market = self.ex.markets[symbol]
        if not self.market.get("spot", False):
            raise ValueError(f"{symbol} is not a spot market; only spot trading is allowed")

    # --- account ----------------------------------------------------------------------
    def _balances(self) -> tuple[float, float]:
        bal = self.ex.fetch_balance()
        units = float((bal.get(self.base) or {}).get("total") or 0.0)
        quote = float((bal.get(self.quote) or {}).get("total") or 0.0)
        return quote, units

    def account(self, price: float) -> tuple[float, float]:
        """Return (cash, units) of the **capped** capital this trader manages.

        Managed equity is ``min(real account value, max_live_notional)``. Any money above
        the cap is invisible to the strategy.
        """
        quote, units = self._balances()
        managed_equity = min(quote + units * price, self.cfg.max_live_notional)
        return managed_equity - units * price, units

    # --- startup checks -----------------------------------------------------------------
    def reconcile(self) -> None:
        """Compare exchange and store state; raise `ReconciliationError` on any mismatch."""
        _, units = self._balances()
        expected = self.store.get("live_units")
        if expected is None:
            self.store.set("live_units", units)
            self.store.event("reconcile", f"baseline {self.base} balance recorded: {units:g}")
        else:
            tol = max(abs(float(expected)), 1e-12) * self.cfg.reconcile_tolerance
            if abs(units - float(expected)) > tol:
                raise ReconciliationError(
                    f"{self.base} balance is {units:g} but the store expects {expected:g}. "
                    "Did you trade manually, or did an order fill while the bot was down? "
                    "Investigate, then delete the 'live_units' state row to accept the new "
                    "baseline."
                )
        for o in self.ex.fetch_open_orders(self.symbol):
            cid = str(o.get("clientOrderId") or "")
            if not cid.startswith(OUR_PREFIX):
                raise ReconciliationError(
                    f"open order {o.get('id')} on {self.symbol} was not placed by windtunnel; "
                    "cancel it by hand before running"
                )

    # --- trading ------------------------------------------------------------------------
    def rebalance(
        self, target_weight: float, price: float, bar_ts: pd.Timestamp, bar_vol: float
    ) -> Fill | None:
        """Submit (or, in shadow mode, log) one limit order towards ``target_weight``.

        It returns None because real fills arrive asynchronously; `sync` records them.
        """
        target_weight = max(0.0, min(1.0, target_weight))  # spot: long/flat, no leverage
        if target_weight == 0.0:
            self._cancel_our_open_orders()  # exiting (incl. kill switch): drop pending buys first
        cash, units = self.account(price)
        slip = self.costs.slippage_bps * BP + self.costs.vol_slippage_k * bar_vol
        plan = plan_fill(
            cash, units, price, target_weight, self.costs.fee_bps * BP, slip, self.rebalance_band
        )
        if plan is None:
            return None
        side = "buy" if plan.qty > 0 else "sell"
        amount = min(abs(plan.qty), units) if side == "sell" else abs(plan.qty)
        cid = client_order_id(self.strategy_name, bar_ts, side)
        if self.store.order(cid) is not None:
            self.store.event("order", f"{cid} already submitted; not resubmitting")
            return None

        band = self.cfg.limit_band_bps * BP
        limit = price * (1 + band) if side == "buy" else price * (1 - band)
        amount = float(self.ex.amount_to_precision(self.symbol, amount))
        limit = float(self.ex.price_to_precision(self.symbol, limit))
        if amount <= 0 or not self._meets_minimums(amount, limit):
            self.store.event("order", f"{side} {amount:g} below exchange minimums; skipped")
            return None
        post_value = (units + amount if side == "buy" else units - amount) * limit
        if side == "buy" and post_value > self.cfg.max_live_notional * 1.001:
            raise CapExceededError(
                f"refusing {side} {amount:g} @ {limit:g}: position value {post_value:.2f} would "
                f"exceed max_live_notional {self.cfg.max_live_notional:g}"
            )
        self._submit(cid, side, amount, limit, bar_ts)
        return None

    def _submit(
        self, cid: str, side: str, amount: float, limit: float, bar_ts: pd.Timestamp
    ) -> None:
        row: dict[str, Any] = {
            "client_id": cid,
            "bar_ts": str(bar_ts),
            "mode": self.mode,
            "side": side,
            "qty": amount,
            "limit_price": limit,
        }
        if self.shadow:
            self.store.record_order(**row, status="shadow")
            self.store.event(
                "shadow", f"WOULD {side} {amount:g} {self.base} @ limit {limit:g} ({cid})"
            )
            return
        # The only order-placing call in the codebase: always a LIMIT order.
        resp = self.ex.create_order(
            self.symbol, "limit", side, amount, limit, {"clientOrderId": cid}
        )
        self.store.record_order(**row, status="open", exchange_id=str(resp.get("id")))
        self.store.event("order", f"placed {side} {amount:g} {self.base} @ {limit:g} ({cid})")

    def _cancel_our_open_orders(self) -> None:
        for o in self.ex.fetch_open_orders(self.symbol):
            if str(o.get("clientOrderId") or "").startswith(OUR_PREFIX):
                if self.shadow:
                    self.store.event("shadow", f"WOULD cancel {o.get('clientOrderId')}")
                else:
                    self.ex.cancel_order(o["id"], self.symbol)
                    self.store.event("order", f"cancelled {o.get('clientOrderId')} before exit")

    def _meets_minimums(self, amount: float, price: float) -> bool:
        limits = self.market.get("limits") or {}
        min_amt = (limits.get("amount") or {}).get("min") or 0.0
        min_cost = (limits.get("cost") or {}).get("min") or 0.0
        return amount >= float(min_amt) and amount * price >= float(min_cost)

    def sync(self) -> None:
        """Record fills of our open orders and cancel any that have timed out."""
        if self.shadow:
            return
        timeout = pd.Timedelta(minutes=self.cfg.order_timeout_minutes)
        now = pd.Timestamp.now(tz="UTC")
        for row in self.store.orders_with_status("open"):
            order = self.ex.fetch_order(row["exchange_id"], self.symbol)
            status = order.get("status")
            if status == "open" and now - pd.Timestamp(str(row["ts"])) > timeout:
                self.ex.cancel_order(row["exchange_id"], self.symbol)
                self.store.event("order", f"cancelled {row['client_id']} after timeout")
                order = self.ex.fetch_order(row["exchange_id"], self.symbol)
                status = "canceled"
            if status in ("closed", "canceled", "expired", "rejected"):
                self._record_final(
                    str(row["client_id"]), str(row["bar_ts"]), str(row["side"]), order, str(status)
                )
        _, units = self._balances()
        self.store.set("live_units", units)

    def _record_final(
        self, cid: str, bar_ts: str, side: str, order: dict[str, Any], status: str
    ) -> None:
        filled = float(order.get("filled") or 0.0)
        with self.store.transaction():
            if filled > 0:
                price = float(order.get("average") or order.get("price") or 0.0)
                fee = float((order.get("fee") or {}).get("cost") or 0.0)
                qty = filled if side == "buy" else -filled
                self.store.record_fill(
                    bar_ts=bar_ts,
                    client_id=cid,
                    qty=qty,
                    price=price,
                    notional=filled * price,
                    fee=fee,
                    slippage=0.0,
                )
            self.store.record_order(
                client_id=cid,
                bar_ts=bar_ts,
                mode=self.mode,
                side=side,
                qty=float(order.get("amount") or filled),
                status=status,
            )
