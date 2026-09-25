"""The live-trading gate and credential loading.

Real orders need ALL THREE of:

1. ``mode = "live"`` in the config file,
2. the environment variable ``WINDTUNNEL_LIVE=I_UNDERSTAND_REAL_MONEY``,
3. the ``--live`` command-line flag.

If any one is missing, the trader runs as the paper trader and logs which one was missing.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LIVE_ENV = "WINDTUNNEL_LIVE"
LIVE_ENV_VALUE = "I_UNDERSTAND_REAL_MONEY"
KEY_ENV = "WINDTUNNEL_API_KEY"
SECRET_ENV = "WINDTUNNEL_API_SECRET"
PASSWORD_ENV = "WINDTUNNEL_API_PASSWORD"  # only some exchanges (e.g. OKX) need this

log = logging.getLogger("windtunnel.live")


@dataclass(frozen=True)
class GateDecision:
    """The effective mode and, if it was downgraded, why."""

    mode: str
    missing: tuple[str, ...] = ()


def resolve_mode(
    config_mode: str, cli_live_flag: bool, env: Mapping[str, str] | None = None
) -> GateDecision:
    """Return the mode the trader will actually run in. It defaults to paper."""
    env = os.environ if env is None else env
    if config_mode == "paper":
        return GateDecision("paper")
    if config_mode == "shadow":
        return GateDecision("shadow")  # real keys, but no orders are ever sent
    missing = []
    if config_mode != "live":
        missing.append('mode = "live" in config')
    if env.get(LIVE_ENV) != LIVE_ENV_VALUE:
        missing.append(f"{LIVE_ENV}={LIVE_ENV_VALUE} in the environment")
    if not cli_live_flag:
        missing.append("--live flag")
    if missing:
        return GateDecision("paper", tuple(missing))
    return GateDecision("live")


class MissingCredentialsError(RuntimeError):
    """Raised when shadow or live mode is requested without API keys in the environment."""


def load_credentials(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return ccxt credential kwargs from environment variables. Never read from a file."""
    env = os.environ if env is None else env
    key, secret = env.get(KEY_ENV), env.get(SECRET_ENV)
    if not key or not secret:
        raise MissingCredentialsError(
            f"set {KEY_ENV} and {SECRET_ENV} in the environment (trade-only key, "
            "withdrawals DISABLED, IP-whitelisted if your exchange allows it)"
        )
    creds: dict[str, Any] = {"apiKey": key, "secret": secret}
    if env.get(PASSWORD_ENV):
        creds["password"] = env[PASSWORD_ENV]
    install_redaction([str(v) for v in creds.values()])
    return creds


class _Redact(logging.Filter):
    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self.secrets = [s for s in secrets if len(s) >= 6]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(s in msg for s in self.secrets):
            for s in self.secrets:
                msg = msg.replace(s, "***REDACTED***")
            record.msg, record.args = msg, None
        return True


def install_redaction(secrets: list[str]) -> None:
    """Scrub these strings from every log record, as defence in depth."""
    filt = _Redact(secrets)
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(filt)
    root.addFilter(filt)


def make_exchange(exchange_id: str, credentials: dict[str, Any]) -> Any:
    """Return an authenticated ccxt exchange, spot market type, with rate limiting on."""
    import ccxt

    ex = getattr(ccxt, exchange_id)(
        {**credentials, "enableRateLimit": True, "options": {"defaultType": "spot"}}
    )
    return ex
