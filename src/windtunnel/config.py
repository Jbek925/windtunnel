"""Configuration for the paper and live trader, loaded from a TOML file (stdlib ``tomllib``).

A config file never contains API keys. Keys come only from environment variables
(see `windtunnel.live.gate`). Any key-like field found in a config file is rejected.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

Mode = Literal["paper", "shadow", "live"]

_FORBIDDEN_KEYS = {"apikey", "api_key", "secret", "api_secret", "password", "private_key", "key"}


@dataclass(frozen=True)
class RiskConfig:
    """Risk limits, enforced by `windtunnel.paper.risk.RiskManager` in every mode."""

    max_weight: float = 1.0
    """Largest allowed position, as a fraction of equity (1.0 = 100%, never leverage)."""
    max_daily_loss: float = 0.05
    """Flatten and pause until the next UTC day if equity falls this much in one day."""
    max_drawdown: float = 0.25
    """Kill switch: flatten and stop for good (until manually reset) at this drawdown."""
    stale_after_bars: float = 1.5
    """Treat data as stale if no new bar arrives within this many bar lengths."""
    max_bar_move: float = 0.25
    """Skip trading on a bar whose close moved more than this vs the previous close."""


@dataclass(frozen=True)
class LiveConfig:
    """Settings used only in ``shadow`` and ``live`` modes."""

    max_live_notional: float = 50.0
    """HARD CAP on position value, in quote currency. Only change this by hand."""
    limit_band_bps: float = 20.0
    """Limit price = reference price ± this band. Never uses market orders."""
    order_timeout_minutes: float = 10.0
    """Cancel unfilled limit orders after this long."""
    reconcile_tolerance: float = 0.02
    """Halt if the exchange balance differs from the recorded position by more than this share."""


@dataclass(frozen=True)
class TraderConfig:
    """Everything the paper/live runner needs, apart from secrets."""

    mode: Mode = "paper"
    exchange: str = "kraken"
    symbol: str = "BTC/USDT"
    timeframe: str = "1d"
    strategy: str = "ma_trend"
    params: dict[str, Any] = field(default_factory=dict)
    sizer: Literal["fixed", "vol"] = "fixed"
    vol_target: float = 0.2
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    rebalance_band: float = 0.02
    initial_equity: float = 1000.0
    """Starting paper capital (quote currency). Ignored in live mode, which uses real balances."""
    history_bars: int = 1000
    """Bars of history fed to the strategy each step. Keep it well above the longest lookback."""
    poll_seconds: float = 60.0
    db_path: str = "data/paper/trader.sqlite"
    log_path: str = "logs/trader.log"
    risk: RiskConfig = field(default_factory=RiskConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def __post_init__(self) -> None:
        if self.mode not in ("paper", "shadow", "live"):
            raise ValueError(f"mode must be paper, shadow or live, got {self.mode!r}")
        if not 0 < self.risk.max_weight <= 1.0:
            raise ValueError("risk.max_weight must be in (0, 1]: spot only, no leverage")
        if self.live.max_live_notional <= 0:
            raise ValueError("live.max_live_notional must be positive")


def _check_no_secrets(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"config key {path + str(k)!r} looks like a credential. API keys must come "
                    "from environment variables only, never from config files."
                )
            _check_no_secrets(v, f"{path}{k}.")


def _build(cls: type[Any], data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: Path | str) -> TraderConfig:
    """Load and validate a `TraderConfig` from a TOML file."""
    raw = tomllib.loads(Path(path).read_text())
    _check_no_secrets(raw)
    risk = _build(RiskConfig, raw.pop("risk", {}))
    live = _build(LiveConfig, raw.pop("live", {}))
    cfg: TraderConfig = _build(TraderConfig, {**raw, "risk": risk, "live": live})
    return cfg
