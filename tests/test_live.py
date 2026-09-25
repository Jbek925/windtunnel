"""Live-trading safeguards, tested against a FAKE exchange. No real account is touched."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from windtunnel.backtest.costs import CostModel
from windtunnel.config import LiveConfig
from windtunnel.live.ccxt_broker import CapExceededError, LiveCcxtBroker, ReconciliationError
from windtunnel.live.gate import (
    LIVE_ENV,
    LIVE_ENV_VALUE,
    MissingCredentialsError,
    load_credentials,
    resolve_mode,
)
from windtunnel.paper.store import Store

SRC = Path(__file__).parents[1] / "src" / "windtunnel"
BAR = pd.Timestamp("2024-05-01", tz="UTC")


class FakeAuthExchange:
    """Mimics the private ccxt calls we allow, records every call."""

    def __init__(self, quote: float = 1000.0, base: float = 0.0) -> None:
        self.balance = {"BTC": base, "USDT": quote}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.orders: dict[str, dict[str, Any]] = {}
        self.markets: dict[str, Any] = {}

    def load_markets(self) -> dict[str, Any]:
        self.markets = {
            "BTC/USDT": {"spot": True, "limits": {"amount": {"min": 1e-5}, "cost": {"min": 5.0}}}
        }
        return self.markets

    def fetch_balance(self) -> dict[str, Any]:
        self.calls.append(("fetch_balance", ()))
        return {k: {"total": v} for k, v in self.balance.items()}

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{amount:.8f}"

    def price_to_precision(self, symbol: str, price: float) -> str:
        return f"{price:.2f}"

    def create_order(
        self, symbol: str, type: str, side: str, amount: float, price: float, params: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(("create_order", (symbol, type, side, amount, price, params)))
        oid = f"ex{len(self.orders)}"
        self.orders[oid] = {
            "id": oid,
            "status": "open",
            "amount": amount,
            "filled": 0.0,
            "price": price,
            "clientOrderId": params["clientOrderId"],
        }
        return self.orders[oid]

    def fill(self, oid: str) -> None:
        o = self.orders[oid]
        o.update(status="closed", filled=o["amount"], average=o["price"], fee={"cost": 0.1})
        self.balance["BTC"] += o["amount"]
        self.balance["USDT"] -= o["amount"] * o["price"] + 0.1

    def fetch_order(self, oid: str, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetch_order", (oid,)))
        return self.orders[oid]

    def cancel_order(self, oid: str, symbol: str) -> None:
        self.calls.append(("cancel_order", (oid,)))
        self.orders[oid]["status"] = "canceled"

    def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return [o for o in self.orders.values() if o["status"] == "open"]


def _broker(
    tmp_path: Path, ex: FakeAuthExchange, *, shadow: bool = False, cap: float = 50.0
) -> LiveCcxtBroker:
    return LiveCcxtBroker(
        ex,
        "BTC/USDT",
        Store(tmp_path / "live.sqlite"),
        CostModel(),
        LiveConfig(max_live_notional=cap),
        strategy_name="ma_trend",
        rebalance_band=0.02,
        shadow=shadow,
    )


# --- the triple opt-in gate --------------------------------------------------------------
ENV_OK = {LIVE_ENV: LIVE_ENV_VALUE}


@pytest.mark.parametrize(
    ("cfg_mode", "flag", "env", "expected"),
    [
        ("live", True, ENV_OK, "live"),  # all three present → live
        ("live", False, ENV_OK, "paper"),  # missing --live
        ("live", True, {}, "paper"),  # missing env var
        ("live", True, {LIVE_ENV: "yes"}, "paper"),  # env var must be the exact phrase
        ("paper", True, ENV_OK, "paper"),  # config says paper
        ("shadow", True, ENV_OK, "shadow"),  # shadow never trades
    ],
)
def test_live_needs_all_three_opt_ins(
    cfg_mode: str, flag: bool, env: dict[str, str], expected: str
) -> None:
    assert resolve_mode(cfg_mode, flag, env).mode == expected


def test_gate_reports_what_is_missing() -> None:
    d = resolve_mode("live", False, {})
    assert d.mode == "paper" and len(d.missing) == 2


def test_credentials_only_from_env_and_redacted(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(MissingCredentialsError):
        load_credentials({})
    creds = load_credentials(
        {"WINDTUNNEL_API_KEY": "KEY123456", "WINDTUNNEL_API_SECRET": "SEC987654"}
    )
    assert creds == {"apiKey": "KEY123456", "secret": "SEC987654"}
    with caplog.at_level(logging.INFO):
        logging.getLogger("windtunnel.test").info("oops leaked SEC987654")
    assert "SEC987654" not in caplog.text


# --- capital cap, limit-only, spot ----------------------------------------------------------
def test_cap_limits_position_to_max_live_notional(tmp_path: Path) -> None:
    ex = FakeAuthExchange(quote=10_000.0)  # far more money than the cap
    b = _broker(tmp_path, ex, cap=50.0)
    b.reconcile()
    b.rebalance(1.0, price=60_000.0, bar_ts=BAR, bar_vol=0.0)
    ((_, (_sym, typ, side, amount, price, params)),) = [
        c for c in ex.calls if c[0] == "create_order"
    ]
    assert typ == "limit" and side == "buy"
    assert amount * price <= 50.0 * 1.001  # never more than the cap, whatever the account holds
    assert params["clientOrderId"].startswith("wt-")


def test_cap_check_refuses_if_sizing_ever_overshoots(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    ex = FakeAuthExchange(quote=10_000.0)
    b = _broker(tmp_path, ex, cap=50.0)
    monkeypatch.setattr(b, "account", lambda price: (5000.0, 0.0))  # simulate a sizing bug
    with pytest.raises(CapExceededError):
        b.rebalance(1.0, price=60_000.0, bar_ts=BAR, bar_vol=0.0)
    assert not [c for c in ex.calls if c[0] == "create_order"]


def test_limit_price_band_and_no_short(tmp_path: Path) -> None:
    ex = FakeAuthExchange(quote=100.0)
    b = _broker(tmp_path, ex)
    b.rebalance(-1.0, price=60_000.0, bar_ts=BAR, bar_vol=0.0)  # short request on spot
    assert not [c for c in ex.calls if c[0] == "create_order"]  # clipped to flat: nothing to do
    b.rebalance(1.0, price=60_000.0, bar_ts=BAR, bar_vol=0.0)
    price = next(c for c in ex.calls if c[0] == "create_order")[1][4]
    assert price == pytest.approx(60_000.0 * 1.002, abs=0.01)  # within the 20 bps band


def test_below_exchange_minimum_is_skipped(tmp_path: Path) -> None:
    ex = FakeAuthExchange(quote=3.0)  # 3 USDT < 5 USDT minimum order
    b = _broker(tmp_path, ex)
    b.rebalance(1.0, price=60_000.0, bar_ts=BAR, bar_vol=0.0)
    assert not [c for c in ex.calls if c[0] == "create_order"]


# --- idempotency, reconciliation, shadow --------------------------------------------------------
def test_same_bar_is_never_submitted_twice_even_after_restart(tmp_path: Path) -> None:
    ex = FakeAuthExchange(quote=1000.0)
    b = _broker(tmp_path, ex)
    b.rebalance(1.0, price=60_000.0, bar_ts=BAR, bar_vol=0.0)
    b.store.close()
    b2 = _broker(tmp_path, ex)  # restart on the same database
    b2.rebalance(1.0, price=60_000.0, bar_ts=BAR, bar_vol=0.0)
    assert len([c for c in ex.calls if c[0] == "create_order"]) == 1


def test_sync_records_fills_and_cancels_timeouts(tmp_path: Path) -> None:
    ex = FakeAuthExchange(quote=1000.0)
    b = _broker(tmp_path, ex)
    b.rebalance(1.0, price=60_000.0, bar_ts=BAR, bar_vol=0.0)
    ex.fill("ex0")
    b.sync()
    fills = b.store.table("fills")
    assert len(fills) == 1 and fills["fee"].iloc[0] == 0.1
    b.rebalance(0.0, price=60_000.0, bar_ts=BAR + pd.Timedelta(days=1), bar_vol=0.0)
    b.store.conn.execute("UPDATE orders SET ts = '2000-01-01T00:00:00+00:00' WHERE status='open'")
    b.sync()
    assert ("cancel_order", ("ex1",)) in ex.calls


def test_reconcile_halts_on_balance_mismatch(tmp_path: Path) -> None:
    ex = FakeAuthExchange(quote=1000.0, base=0.001)
    b = _broker(tmp_path, ex)
    b.reconcile()  # first run records the baseline
    ex.balance["BTC"] = 0.005  # someone traded by hand
    with pytest.raises(ReconciliationError, match="balance"):
        _broker(tmp_path, ex).reconcile()


def test_reconcile_halts_on_foreign_open_orders(tmp_path: Path) -> None:
    ex = FakeAuthExchange()
    ex.orders["manual"] = {"id": "manual", "status": "open", "clientOrderId": "my-own-order"}
    with pytest.raises(ReconciliationError, match="not placed by windtunnel"):
        _broker(tmp_path, ex).reconcile()


def test_shadow_mode_never_sends_or_cancels(tmp_path: Path) -> None:
    ex = FakeAuthExchange(quote=1000.0)
    b = _broker(tmp_path, ex, shadow=True)
    b.rebalance(1.0, price=60_000.0, bar_ts=BAR, bar_vol=0.0)
    b.rebalance(0.0, price=60_000.0, bar_ts=BAR + pd.Timedelta(days=1), bar_vol=0.0)
    assert not [c for c in ex.calls if c[0] in ("create_order", "cancel_order")]
    assert (b.store.table("orders")["status"] == "shadow").all()


def test_non_spot_market_rejected(tmp_path: Path) -> None:
    ex = FakeAuthExchange()
    ex.load_markets = lambda: setattr(ex, "markets", {"BTC/USDT": {"spot": False}})  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="spot"):
        _broker(tmp_path, ex)


# --- static checks over the whole source tree --------------------------------------------------
PRIVATE_ALLOWED = {
    "fetch_balance",
    "create_order",
    "fetch_order",
    "cancel_order",
    "fetch_open_orders",
}
PRIVATE_FORBIDDEN = {
    "withdraw",
    "transfer",
    "create_market_order",
    "create_market_buy_order",
    "create_market_sell_order",
    "set_leverage",
    "set_margin_mode",
    "borrow_margin",
    "repay_margin",
    "edit_order",
    "create_stop_order",
    "fetch_deposit_address",
    "create_limit_order",
    "create_order_ws",
    "cancel_all_orders",
    "set_position_mode",
}


def _calls() -> list[tuple[Path, ast.Call, str]]:
    out = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                out.append((path, node, node.func.attr))
    return out


def test_no_forbidden_exchange_methods_anywhere() -> None:
    bad = [(p.name, a) for p, _, a in _calls() if a in PRIVATE_FORBIDDEN]
    assert bad == []


def test_private_methods_only_in_live_package() -> None:
    live = SRC / "live"
    bad = [
        (str(p.relative_to(SRC)), a)
        for p, _, a in _calls()
        if a in PRIVATE_ALLOWED and live not in p.parents
    ]
    assert bad == []


def test_every_create_order_is_a_limit_order() -> None:
    orders = [n for _, n, a in _calls() if a == "create_order"]
    assert len(orders) == 1
    assert isinstance(orders[0].args[1], ast.Constant) and orders[0].args[1].value == "limit"


def test_credentials_only_referenced_in_live_package() -> None:
    for path in SRC.rglob("*.py"):
        if (SRC / "live") in path.parents:
            continue
        text = path.read_text()
        assert "WINDTUNNEL_API_KEY" not in text and "WINDTUNNEL_API_SECRET" not in text, path
        assert "apiKey" not in text, path
