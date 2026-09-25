"""SQLite persistence for the paper/live trader.

Everything needed to restart safely lives here: account state, the last processed bar,
risk latches, and a full log of orders, fills, equity and events. Writes that belong
together happen in one transaction, so a crash can never leave a fill recorded without
the bar being marked as processed (or the other way round).

Secrets are never stored here.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, level TEXT NOT NULL,
    kind TEXT NOT NULL, message TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS orders (
    client_id TEXT PRIMARY KEY, ts TEXT NOT NULL, bar_ts TEXT NOT NULL, mode TEXT NOT NULL,
    side TEXT NOT NULL, qty REAL NOT NULL, limit_price REAL, status TEXT NOT NULL,
    exchange_id TEXT);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, bar_ts TEXT NOT NULL, client_id TEXT,
    qty REAL NOT NULL, price REAL NOT NULL, notional REAL NOT NULL, fee REAL NOT NULL,
    slippage REAL NOT NULL, weight_before REAL, weight_after REAL);
CREATE TABLE IF NOT EXISTS equity (
    bar_ts TEXT PRIMARY KEY, ts TEXT NOT NULL, close REAL NOT NULL, cash REAL NOT NULL,
    units REAL NOT NULL, equity REAL NOT NULL, target_weight REAL);
"""


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


class Store:
    """A thin wrapper around one SQLite file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), isolation_level=None)  # explicit transactions
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._depth = 0

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group writes atomically. Nested use joins the outer transaction."""
        if self._depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("ROLLBACK")
            raise
        self._depth -= 1
        if self._depth == 0:
            self.conn.execute("COMMIT")

    # --- key/value state -------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return a JSON-decoded state value, or ``default``."""
        row = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def set(self, key: str, value: Any) -> None:
        """Store a JSON-encodable state value."""
        self.conn.execute(
            "INSERT INTO state(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    # --- logs ------------------------------------------------------------------------
    def event(self, kind: str, message: str, level: str = "INFO") -> None:
        """Append to the event log (risk triggers, restarts, stale data, errors...)."""
        self.conn.execute(
            "INSERT INTO events(ts, level, kind, message) VALUES (?, ?, ?, ?)",
            (_now(), level, kind, message),
        )

    def record_order(self, **row: Any) -> None:
        """Insert or update an order row, keyed by ``client_id``."""
        row.setdefault("ts", _now())
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        updates = ", ".join(f"{c} = excluded.{c}" for c in row if c != "client_id")
        self.conn.execute(
            f"INSERT INTO orders({cols}) VALUES ({marks}) "
            f"ON CONFLICT(client_id) DO UPDATE SET {updates}",
            tuple(row.values()),
        )

    def order(self, client_id: str) -> dict[str, Any] | None:
        """Return an order row by client id."""
        row = self.conn.execute("SELECT * FROM orders WHERE client_id = ?", (client_id,)).fetchone()
        return None if row is None else dict(row)

    def orders_with_status(self, status: str) -> list[dict[str, Any]]:
        """Return all order rows with the given status."""
        rows = self.conn.execute("SELECT * FROM orders WHERE status = ?", (status,)).fetchall()
        return [dict(r) for r in rows]

    def record_fill(self, **row: Any) -> None:
        """Append a fill."""
        row.setdefault("ts", _now())
        cols = ", ".join(row)
        self.conn.execute(
            f"INSERT INTO fills({cols}) VALUES ({', '.join('?' for _ in row)})",
            tuple(row.values()),
        )

    def record_equity(self, **row: Any) -> None:
        """Insert (or replace) the equity snapshot for one bar."""
        row.setdefault("ts", _now())
        cols = ", ".join(row)
        self.conn.execute(
            f"INSERT OR REPLACE INTO equity({cols}) VALUES ({', '.join('?' for _ in row)})",
            tuple(row.values()),
        )

    def table(self, name: str) -> pd.DataFrame:
        """Return a whole table as a DataFrame (for status and comparison)."""
        if name not in {"events", "orders", "fills", "equity", "state"}:
            raise ValueError(name)
        return pd.read_sql_query(f"SELECT * FROM {name}", self.conn)
