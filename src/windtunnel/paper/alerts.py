"""Alerts: always logged and stored; optionally POSTed to a webhook (e.g. ntfy, Discord, Slack).

The webhook URL comes from the environment variable ``WINDTUNNEL_ALERT_WEBHOOK``, never
from config. The body is JSON ``{"text": "..."}``, which Discord and Slack accept.
For ntfy.sh, use ``https://ntfy.sh/<your-topic>``. It accepts any body.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

from windtunnel.paper.store import Store

log = logging.getLogger("windtunnel.alerts")
WEBHOOK_ENV = "WINDTUNNEL_ALERT_WEBHOOK"


class Alerter:
    """Record an event, log it, and optionally push it to a webhook."""

    def __init__(self, store: Store, prefix: str = "windtunnel") -> None:
        self.store = store
        self.prefix = prefix
        self.webhook = os.environ.get(WEBHOOK_ENV) or None

    def __call__(self, kind: str, message: str, level: str = "WARNING") -> None:
        """Send one alert. Delivery failures are logged but never crash the trader."""
        self.store.event(kind, message, level)
        log.log(logging.getLevelName(level), "%s: %s", kind, message)
        if self.webhook and level in ("WARNING", "ERROR", "CRITICAL"):
            body = json.dumps({"text": f"[{self.prefix}] {level} {kind}: {message}"}).encode()
            req = urllib.request.Request(
                self.webhook, data=body, headers={"Content-Type": "application/json"}
            )
            try:
                urllib.request.urlopen(req, timeout=10).close()
            except Exception as exc:  # alerts are best-effort
                log.error("alert webhook failed: %s", exc)
