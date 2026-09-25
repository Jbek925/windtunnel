"""Placeholder for a read-only prediction-market data adapter. It will be filled in later.

Intended design, when this is implemented:

* **Read-only public endpoints only**, for example public market-history APIs from
  prediction-market venues. No accounts, keys or order placement.
* The data is a *probability* series in [0, 1] per contract, not OHLCV prices. It will
  probably need its own schema, e.g. ``timestamp`` (UTC) plus ``price_yes`` and
  ``volume``, and cannot reuse the OHLCV strategies unchanged.
* Contracts resolve to 0 or 1 on a fixed date. That makes the survivorship and
  lookahead traps different from spot markets. The resolution outcome must never
  leak into features.
"""

from __future__ import annotations

import pandas as pd

from windtunnel.data.schema import Calendar


class PredictionMarketSource:
    """Stub `DataSource` for prediction markets. Every fetch raises `NotImplementedError`."""

    name = "prediction_market"
    calendar: Calendar = "24/7"

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Not implemented yet. See the module docstring for the intended design."""
        raise NotImplementedError(
            "prediction-market data is a stub; see windtunnel.data.prediction_stub docstring"
        )

    def supported_timeframes(self) -> set[str]:
        """Return an empty set, because nothing is implemented yet."""
        return set()
