"""Robustness and honesty checks: how much should we believe a backtest?

* `bootstrap_sharpe_ci`: resamples blocks of returns to give a confidence interval
  for the Sharpe ratio. A CI that straddles 0 means "can't tell this apart from luck".
* `deflated_sharpe_ratio`: Bailey & López de Prado (2014). If you try N strategies,
  the best one's Sharpe is inflated by selection alone. DSR is the probability that the
  true Sharpe is above what the best of N *useless* strategies would show by chance.
* `parameter_sensitivity`: Sharpe across the whole parameter grid. A good result that
  only exists at one sharp peak is probably noise. Broad plateaus are more believable.
* `verdict`: turns all of the above into plain-English sentences, without flattery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from windtunnel.backtest.costs import CostModel
from windtunnel.backtest.engine import run_backtest
from windtunnel.backtest.sizing import Sizer
from windtunnel.evaluation.metrics import Metrics, sharpe_ratio
from windtunnel.strategies.base import Strategy

EULER_GAMMA = 0.5772156649015329


@dataclass
class SharpeCI:
    """A point estimate and bootstrap confidence interval for an annualised Sharpe ratio."""

    sharpe: float
    low: float
    high: float
    level: float

    @property
    def excludes_zero(self) -> bool:
        """Return True if the whole interval is above zero."""
        return self.low > 0


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Return an ``(n_boot, n)`` array of resampled indices (Politis & Romano, 1994).

    Blocks start at random positions and have geometric lengths with mean ``mean_block``,
    wrapping around the end. Keeping blocks together preserves short-range dependence
    such as volatility clustering, which a plain i.i.d. bootstrap would destroy.
    """
    new_block = rng.random((n_boot, n)) < 1.0 / mean_block
    new_block[:, 0] = True
    block_id = np.cumsum(new_block, axis=1) - 1
    n_blocks = int(block_id.max()) + 1
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    t = np.broadcast_to(np.arange(n), (n_boot, n))
    first_t = np.maximum.accumulate(np.where(new_block, t, 0), axis=1)
    offset = t - first_t
    start_of_block = np.take_along_axis(starts, block_id, axis=1)
    idx: np.ndarray = (start_of_block + offset) % n
    return idx


def bootstrap_sharpe_ci(
    returns: pd.Series,
    periods_per_year: float,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    mean_block: float | None = None,
    seed: int = 0,
) -> SharpeCI:
    """Return a stationary-bootstrap confidence interval for the annualised Sharpe ratio."""
    r = returns.dropna().to_numpy()
    n = len(r)
    point = sharpe_ratio(returns, periods_per_year)
    if n < 10:
        return SharpeCI(point, float("nan"), float("nan"), level)
    block = mean_block or max(5.0, round(n ** (1 / 3)))
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_boot)
    batch = max(1, min(n_boot, 2_000_000 // n))  # bound memory on long hourly series
    for b0 in range(0, n_boot, batch):
        k = min(batch, n_boot - b0)
        sample = r[stationary_bootstrap_indices(n, k, block, rng)]
        sd = sample.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(sd > 0, sample.mean(axis=1) / sd, 0.0)
        sharpes[b0 : b0 + k] = sr * np.sqrt(periods_per_year)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(sharpes, [alpha, 1 - alpha])
    return SharpeCI(point, float(lo), float(hi), level)


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Return P(true per-bar Sharpe > ``sr_benchmark``), allowing for skew and fat tails.

    This is Bailey & López de Prado (2012). Both Sharpes here are **per bar**, not annualised.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # normal = 3
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Return the expected best per-bar Sharpe among ``n_trials`` skill-less strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(trial_sharpe_std * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


@dataclass
class DeflatedSharpe:
    """The deflated Sharpe ratio and the inputs that produced it."""

    n_trials: int
    sr_threshold_annual: float
    """The annualised Sharpe the best of ``n_trials`` random strategies would reach by luck."""
    psr_vs_zero: float
    dsr: float
    """Probability that the true Sharpe beats the luck threshold. > 0.95 is convincing."""


def deflated_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: float,
    *,
    n_trials: int,
    trial_sharpes_annual: list[float] | None = None,
) -> DeflatedSharpe:
    """Return the deflated Sharpe ratio of ``returns`` given ``n_trials`` configurations tried.

    ``trial_sharpes_annual`` should hold the Sharpe of each configuration tried. Their
    spread sets how far luck alone could push the best one. If they aren't given, the
    spread defaults to the sampling error of a Sharpe over this many bars, 1/sqrt(T).
    """
    n = len(returns.dropna())
    if trial_sharpes_annual and len(trial_sharpes_annual) > 1:
        sd_per_bar = float(np.std(trial_sharpes_annual, ddof=1)) / np.sqrt(periods_per_year)
    else:
        sd_per_bar = 1.0 / np.sqrt(max(n, 2))
    sr0 = expected_max_sharpe(max(n_trials, 1), sd_per_bar)
    return DeflatedSharpe(
        n_trials=n_trials,
        sr_threshold_annual=sr0 * float(np.sqrt(periods_per_year)),
        psr_vs_zero=probabilistic_sharpe_ratio(returns, 0.0),
        dsr=probabilistic_sharpe_ratio(returns, sr0),
    )


def parameter_sensitivity(
    bars: pd.DataFrame,
    strategy_cls: type[Strategy],
    sizer: Sizer,
    costs: CostModel,
    *,
    periods_per_year: float,
    long_only: bool = True,
) -> pd.DataFrame:
    """Return the Sharpe (after costs) of every grid configuration over all of ``bars``.

    This is **in-sample**, over the full period, so it shows the *shape* of the
    parameter landscape, not achievable performance. Look for plateaus, not peaks.
    """
    rows: list[dict[str, Any]] = []
    for params in strategy_cls.grid() or [{}]:
        res = run_backtest(
            bars,
            strategy_cls(**params),
            sizer,
            costs,
            periods_per_year=periods_per_year,
            long_only=long_only,
        )
        rows.append(
            {
                **params,
                "sharpe": sharpe_ratio(res.returns, periods_per_year),
                "n_trades": res.n_trades,
            }
        )
    return pd.DataFrame(rows)


def verdict(
    strategy: Metrics,
    benchmarks: list[Metrics],
    ci: SharpeCI,
    dsr: DeflatedSharpe,
    strategy_costs_x2: Metrics | None = None,
) -> list[str]:
    """Return plain-English conclusions, each one a sentence. No flattery."""
    lines: list[str] = []
    bh = benchmarks[0]
    beats_sharpe = [b.label for b in benchmarks if strategy.sharpe > b.sharpe]
    beats_cagr = [b.label for b in benchmarks if strategy.cagr > b.cagr]
    if len(beats_sharpe) == len(benchmarks) and ci.excludes_zero and dsr.dsr > 0.95:
        lines.append(
            f"Out of sample and after costs, the strategy's Sharpe ({strategy.sharpe:.2f}) beat "
            f"every benchmark, its {ci.level:.0%} CI excludes zero, and it survives the "
            f"multiple-testing adjustment (DSR {dsr.dsr:.2f}). This is still one historical "
            "path: paper-trade before trusting it."
        )
    elif not beats_sharpe:
        lines.append(
            f"The strategy did NOT beat buy-and-hold after costs out of sample: Sharpe "
            f"{strategy.sharpe:.2f} vs {bh.sharpe:.2f} (buy & hold). There is no evidence "
            "of an edge here."
        )
    else:
        lines.append(
            f"Mixed result: out-of-sample Sharpe {strategy.sharpe:.2f} beat "
            f"{', '.join(beats_sharpe)} but the evidence is weak. Treat as no proven edge."
        )
    lines.append(
        f"CAGR {strategy.cagr:.1%} vs buy & hold {bh.cagr:.1%}"
        + ("." if beats_cagr else ": lower return than simply holding.")
        + f" Max drawdown {strategy.max_drawdown:.0%} vs {bh.max_drawdown:.0%}."
    )
    if not ci.excludes_zero:
        lines.append(
            f"The {ci.level:.0%} bootstrap CI for Sharpe is [{ci.low:.2f}, {ci.high:.2f}], "
            "which includes zero, so the result is statistically indistinguishable from luck."
        )
    lines.append(
        f"{dsr.n_trials} configuration(s) were tried; the best of that many *useless* "
        f"strategies would be expected to show a Sharpe of about {dsr.sr_threshold_annual:.2f} "
        f"by luck. Deflated Sharpe = {dsr.dsr:.2f} (want > 0.95). If you tried other "
        "strategies, symbols or timeframes before this one, the true count is higher and "
        "this number is optimistic."
    )
    if strategy_costs_x2 is not None:
        lines.append(
            f"With costs doubled, Sharpe falls to {strategy_costs_x2.sharpe:.2f} "
            f"(CAGR {strategy_costs_x2.cagr:.1%}); costs took {strategy.cost_drag:.1%} a year "
            f"at the base assumption, with {strategy.turnover:.1f}× annual turnover."
        )
    return lines
