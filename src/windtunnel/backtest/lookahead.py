"""Lookahead detection by truncation invariance.

If a function of bar history is causal, the value it gives at bar t cannot change when
we delete every bar after t. So for random cut points t we compare::

    f(bars.iloc[: t + 1]).iloc[-1]   # only the past is available
    f(bars).iloc[t]                  # the full history is available

Any difference proves that ``f`` used future data. This catches ``shift(-k)``, centred
rolling windows, full-sample normalisation (``x / x.max()``), ``bfill`` and similar.

What it **cannot** catch is leakage baked into the *data* itself, such as dividend-adjusted
prices that are revised after the fact, or survivorship in the choice of symbols.
Those need judgment, not a test.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from windtunnel.strategies.base import Strategy


class LookaheadError(AssertionError):
    """Raised when a function's output at bar t depends on bars after t."""


def check_causal(
    fn: Callable[[pd.DataFrame], pd.Series],
    bars: pd.DataFrame,
    *,
    n_checks: int = 40,
    min_history: int = 50,
    seed: int = 0,
    atol: float = 1e-9,
    label: str = "function",
) -> None:
    """Raise `LookaheadError` if ``fn`` is not truncation-invariant on ``bars``.

    Args:
        fn: maps a bar frame to a Series on the same index.
        bars: test data. It should be long enough for ``fn`` to warm up.
        n_checks: number of random cut points (the last bar is always included).
        min_history: the earliest cut point.
        seed: RNG seed for choosing cut points.
        atol: absolute tolerance, for floating-point noise from rolling sums.
        label: name used in the error message.
    """
    n = len(bars)
    if n <= min_history + 1:
        raise ValueError("not enough bars to check for lookahead")
    full = fn(bars)
    rng = np.random.default_rng(seed)
    cuts = rng.choice(np.arange(min_history, n - 1), size=min(n_checks, n - 1 - min_history),
                      replace=False)  # fmt: skip
    bad: list[str] = []
    for t in sorted(int(x) for x in cuts):
        truncated = float(fn(bars.iloc[: t + 1]).iloc[-1])
        seen = float(full.iloc[t])
        same = (np.isnan(truncated) and np.isnan(seen)) or abs(truncated - seen) <= atol
        if not same:
            bad.append(f"t={bars.index[t]}: truncated={truncated:.6g} full={seen:.6g}")
    if bad:
        raise LookaheadError(
            f"{label} uses future data at {len(bad)}/{len(cuts)} checked bars, e.g.\n  "
            + "\n  ".join(bad[:5])
        )


def check_no_lookahead(
    strategy: Strategy,
    bars: pd.DataFrame,
    *,
    n_checks: int = 40,
    min_history: int = 50,
    seed: int = 0,
) -> None:
    """Raise `LookaheadError` if ``strategy.signal`` peeks at future bars."""
    check_causal(
        strategy.signal,
        bars,
        n_checks=n_checks,
        min_history=min_history,
        seed=seed,
        label=repr(strategy),
    )
