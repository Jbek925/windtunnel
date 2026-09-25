"""Transaction-cost model shared by backtest, paper and live trading.

Every cost is expressed as a fraction of the **notional traded** (fees, slippage) or of
the **notional held** (funding). 1 bp = 0.01% = 0.0001.

* **fees:** the exchange's maker or taker fee. A strategy that trades at the open with
  limit orders *might* earn maker rates, but assuming taker is the honest default.
* **fixed slippage:** the half-spread plus market impact, in bps per unit traded.
* **volatility-scaled slippage:** ``vol_slippage_k × σ_bar`` extra, where ``σ_bar`` is
  the trailing per-bar volatility known *before* the fill. Spreads widen when markets
  are volatile, which is also when trend strategies tend to trade.
* **funding:** for perpetual futures only. It is charged on the held notional every 8h.
  A positive rate means longs pay. Spot trading has no funding, so the default is 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

BP = 1e-4
_EIGHT_HOURS = pd.Timedelta(hours=8)


@dataclass(frozen=True)
class CostModel:
    """Configurable trading costs. The defaults are roughly a retail crypto spot account.

    Attributes:
        taker_fee_bps: fee for orders that take liquidity.
        maker_fee_bps: fee for resting limit orders that are filled.
        use_maker: charge maker instead of taker fees (optimistic, so off by default).
        slippage_bps: fixed slippage per unit notional traded.
        vol_slippage_k: extra slippage as a multiple of the trailing per-bar volatility.
        vol_lookback: bars used to estimate that volatility.
        funding_rate_per_8h: perp funding rate per 8h (0.0001 = 1 bp). 0 for spot.
    """

    taker_fee_bps: float = 10.0
    maker_fee_bps: float = 10.0
    use_maker: bool = False
    slippage_bps: float = 2.0
    vol_slippage_k: float = 0.0
    vol_lookback: int = 20
    funding_rate_per_8h: float = 0.0

    def __post_init__(self) -> None:
        for name in ("taker_fee_bps", "maker_fee_bps", "slippage_bps", "vol_slippage_k"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")

    @property
    def fee_bps(self) -> float:
        """The fee rate actually charged, in bps."""
        return self.maker_fee_bps if self.use_maker else self.taker_fee_bps

    def trailing_bar_vol(self, bars: pd.DataFrame) -> pd.Series:
        """Per-bar volatility known at the **open** of each bar.

        This is the std of close-to-close log returns up to close t-1, aligned to bar t.
        Early bars without enough history get 0, so only the fixed costs apply there.
        """
        logret = pd.Series(np.log(bars["close"].to_numpy()), index=bars.index).diff()
        vol = logret.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(1)
        return vol.fillna(0.0)

    def next_bar_vol(self, completed_bars: pd.DataFrame) -> float:
        """Return the per-bar vol known at the open of the bar *after* ``completed_bars``.

        Live equivalent of ``trailing_bar_vol(...)`` at the next index: the std of the
        last ``vol_lookback`` close-to-close log returns. Returns 0 if history is too short.
        """
        logret = np.diff(np.log(completed_bars["close"].to_numpy()))
        if len(logret) < self.vol_lookback:
            return 0.0
        return float(np.std(logret[-self.vol_lookback :], ddof=1))

    def cost_rate(self, bar_vol: pd.Series | float = 0.0) -> pd.Series | float:
        """Return the cost per unit of notional traded (a fraction, not bps).

        The rate is fee + fixed slippage + ``vol_slippage_k × bar_vol``.
        """
        return (self.fee_bps + self.slippage_bps) * BP + self.vol_slippage_k * bar_vol

    def funding_rate_per_bar(self, bar_length: pd.Timedelta) -> float:
        """Return the funding charged on held notional for one bar."""
        return self.funding_rate_per_8h * (bar_length / _EIGHT_HOURS)

    def scaled(self, factor: float) -> CostModel:
        """Return a copy with fees and slippage multiplied by ``factor``.

        This is used for the "costs doubled" robustness check. Funding is a published
        rate, not an estimate, so it is left unchanged.
        """
        return replace(
            self,
            taker_fee_bps=self.taker_fee_bps * factor,
            maker_fee_bps=self.maker_fee_bps * factor,
            slippage_bps=self.slippage_bps * factor,
            vol_slippage_k=self.vol_slippage_k * factor,
        )


ZERO_COSTS = CostModel(taker_fee_bps=0.0, maker_fee_bps=0.0, slippage_bps=0.0)
"""A frictionless cost model. Use it only to measure gross performance and cost drag."""
