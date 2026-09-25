"""The common `Strategy` interface."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd


class Strategy(ABC):
    """A rule mapping bar history to a desired exposure in [-1, 1].

    Contract (checked by `windtunnel.backtest.lookahead.check_no_lookahead`):

    * ``signal(bars)`` returns a Series on ``bars.index``.
    * The value at bar t depends **only** on ``bars`` up to and including bar t.
      In practice ``signal(bars.iloc[:t+1]).iloc[-1] == signal(bars).iloc[t]``.
    * Strategies never shift their own output to "trade next bar". The engine does
      that in exactly one place.
    * NaN means "no opinion yet" (warm-up), and the engine treats it as flat.

    Subclasses declare ``param_grid``: the small, explicit set of values that walk-forward
    tuning may try. Every entry counts toward the multiple-testing trial count, so keep
    it small.
    """

    name: ClassVar[str] = "strategy"
    param_grid: ClassVar[dict[str, list[Any]]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.param_grid)
        if unknown:
            raise TypeError(f"{type(self).__name__} got unknown params {sorted(unknown)}")
        defaults = {k: v[0] for k, v in self.param_grid.items()}
        self.params: dict[str, Any] = {**defaults, **params}

    @abstractmethod
    def signal(self, bars: pd.DataFrame) -> pd.Series:
        """Return the desired exposure in [-1, 1] at each bar close."""

    @classmethod
    def grid(cls) -> list[dict[str, Any]]:
        """Return every parameter combination in ``param_grid``."""
        keys = list(cls.param_grid)
        return [
            dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*cls.param_grid.values())
        ]

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"
