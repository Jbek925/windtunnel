"""Benchmarks. Every strategy result is shown next to these.

* **Buy and hold:** 100% invested from the first possible fill. It pays one entry fee,
  and that's it.
* **Vol-targeted buy and hold:** always long, but sized by `VolTarget`, so it holds less
  when the market is wild. That makes it a fairer comparison for vol-targeted strategies,
  since much of their "edge" is often just the vol targeting.

If a strategy can't beat these **after costs, out of sample**, it isn't worth trading.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from windtunnel.backtest.sizing import FixedFraction, Sizer, VolTarget
from windtunnel.strategies.base import Strategy


class BuyAndHold(Strategy):
    """Always fully long. No parameters, so it adds nothing to the trial count."""

    name = "buy_and_hold"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        """Return +1 on every bar."""
        return pd.Series(1.0, index=bars.index)


@dataclass(frozen=True)
class Benchmark:
    """A named (strategy, sizer) pair used as a comparison baseline."""

    label: str
    strategy: Strategy
    sizer: Sizer


def standard_benchmarks(vol_target: VolTarget | None = None) -> list[Benchmark]:
    """Return plain buy-and-hold and vol-targeted buy-and-hold.

    Pass the same `VolTarget` the strategy uses, so the comparison is like-for-like.
    """
    vt = vol_target or VolTarget()
    return [
        Benchmark("buy & hold", BuyAndHold(), FixedFraction(1.0)),
        Benchmark(f"vol-target B&H ({vt.target_ann_vol:.0%})", BuyAndHold(), vt),
    ]
