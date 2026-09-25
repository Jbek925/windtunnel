"""Trading strategies. All implement `windtunnel.strategies.base.Strategy`."""

from windtunnel.strategies.base import Strategy
from windtunnel.strategies.benchmark import Benchmark, BuyAndHold, standard_benchmarks
from windtunnel.strategies.mean_reversion import ZScoreMeanReversion
from windtunnel.strategies.trend import MovingAverageTrend, TimeSeriesMomentum

STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (MovingAverageTrend, TimeSeriesMomentum, ZScoreMeanReversion, BuyAndHold)
}
"""Registry used by the CLI: strategy name → class."""

__all__ = [
    "STRATEGIES",
    "Benchmark",
    "BuyAndHold",
    "MovingAverageTrend",
    "Strategy",
    "TimeSeriesMomentum",
    "ZScoreMeanReversion",
    "standard_benchmarks",
]
