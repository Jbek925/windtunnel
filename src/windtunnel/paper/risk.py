"""Risk controls, enforced in paper, shadow and live modes alike.

Checked once per completed bar, *after* the strategy has proposed a target weight:

1. **Kill switch (latched):** if equity falls ``max_drawdown`` below its peak, go flat and
   stay flat, even across restarts, until someone runs ``paper run --reset-kill-switch``.
2. **Daily loss:** if equity falls ``max_daily_loss`` below the day's opening equity (UTC),
   go flat for the rest of that UTC day.
3. **Price sanity:** if the latest close moved more than ``max_bar_move`` from the previous
   close, don't trade on it. It may be a bad print, so hold the current position and alert.
4. **Max position:** clip the target to ``max_weight`` (≤ 1, never leverage).

Stale-data detection lives in `windtunnel.paper.feed.is_stale`, because it has to run
before any of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from windtunnel.config import RiskConfig
from windtunnel.paper.store import Store


@dataclass
class RiskDecision:
    """What the runner is allowed to do this bar."""

    target: float
    can_trade: bool = True
    reasons: list[str] = field(default_factory=list)


class RiskManager:
    """Applies `RiskConfig` limits. Its latches and peaks persist in the `Store`."""

    def __init__(self, cfg: RiskConfig, store: Store) -> None:
        self.cfg = cfg
        self.store = store

    @property
    def killed(self) -> bool:
        """Return True while the drawdown kill switch is latched."""
        return bool(self.store.get("kill_switch", False))

    def reset_kill_switch(self) -> None:
        """Manually clear the kill switch and restart peak tracking from current equity."""
        self.store.set("kill_switch", False)
        self.store.set("peak_equity", None)
        self.store.event("risk", "kill switch manually reset", "WARNING")

    def evaluate(
        self,
        bar_close_time: pd.Timestamp,
        equity: float,
        close: float,
        prev_close: float | None,
        proposed: float,
    ) -> RiskDecision:
        """Return the risk-adjusted decision for a proposed target weight."""
        d = RiskDecision(target=max(-self.cfg.max_weight, min(self.cfg.max_weight, proposed)))
        if d.target != proposed:
            d.reasons.append(f"target {proposed:.3f} clipped to max_weight {self.cfg.max_weight}")

        peak = max(float(self.store.get("peak_equity") or equity), equity)
        self.store.set("peak_equity", peak)
        dd = equity / peak - 1.0
        if not self.killed and dd <= -self.cfg.max_drawdown:
            self.store.set("kill_switch", True)
            d.reasons.append(
                f"KILL SWITCH: drawdown {dd:.1%} breached -{self.cfg.max_drawdown:.0%}; "
                "flattening and stopping until manually reset"
            )
        if self.killed:
            d.target = 0.0
            if not d.reasons:
                d.reasons.append("kill switch is latched; staying flat")
            return d

        day = str(bar_close_time.tz_convert("UTC").date())
        day_state = self.store.get("day_start") or {}
        if day_state.get("day") != day:
            day_state = {"day": day, "equity": equity}
            self.store.set("day_start", day_state)
        day_ret = equity / float(day_state["equity"]) - 1.0
        if day_ret <= -self.cfg.max_daily_loss and self.store.get("paused_day") != day:
            self.store.set("paused_day", day)
            d.reasons.append(
                f"daily loss {day_ret:.1%} breached -{self.cfg.max_daily_loss:.0%}; "
                f"flat until the next UTC day"
            )
        if self.store.get("paused_day") == day:
            d.target = 0.0
            return d

        if prev_close is not None and prev_close > 0:
            move = close / prev_close - 1.0
            if abs(move) > self.cfg.max_bar_move:
                d.can_trade = False
                d.reasons.append(
                    f"price moved {move:+.1%} in one bar (limit {self.cfg.max_bar_move:.0%}); "
                    "possible bad data, not trading on it"
                )
        return d
