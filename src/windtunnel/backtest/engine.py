"""The backtest engine: the single place where signals become trades.

Timing contract (see CLAUDE.md), for each bar ``i``:

1. At **open[i]**, trade from the current holding to the target weight decided at
   **close[i-1]**. This is the only "shift" in the whole codebase. The fill price is
   open[i]. Fees and slippage are charged on the notional traded.
2. Hold those units through the bar. Funding (perps only) is charged on held notional.
3. Mark to market at **close[i]**. The strategy then sees bar i and sets the next target.

The engine tracks **units and cash**, not just weights, so:

* the close[i-1] → open[i] gap accrues to the position held *before* the trade;
* weights drift with prices between rebalances. Bringing them back to target is a real
  trade that is really charged, but only when the gap exceeds ``rebalance_band``
  (default 2% of equity). That avoids paying fees every bar to chase tiny drifts.
  Moves **to flat** always execute, whatever the band.

The loop is plain Python on purpose. It is easy to read, exactly mirrors what the paper
and live traders do, and is fast enough (about 1 s per 100k bars).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from windtunnel.backtest.costs import BP, CostModel
from windtunnel.backtest.sizing import Sizer
from windtunnel.data.schema import validate_schema
from windtunnel.strategies.base import Strategy


@dataclass
class BacktestResult:
    """The output of a backtest. All series are indexed by bar open time (UTC)."""

    equity: pd.Series
    """Equity at each bar close (starts at ``initial_equity`` before the first bar)."""
    returns: pd.Series
    """Simple returns of equity, bar by bar, net of all costs."""
    weights: pd.Series
    """The weight actually held during each bar (after the open trade)."""
    target_weights: pd.Series
    """The target weight decided at each bar's close, which is filled at the next open."""
    trades: pd.DataFrame
    """One row per fill: timestamp, qty, price, notional, fee, slippage, weight before and after."""
    costs: pd.DataFrame
    """Per-bar ``fees``, ``slippage`` and ``funding``, in quote currency."""
    periods_per_year: float
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        """Return the number of fills."""
        return len(self.trades)

    @property
    def total_costs(self) -> float:
        """Return all costs paid, in quote currency."""
        return float(self.costs.to_numpy().sum())


def run_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    sizer: Sizer,
    costs: CostModel,
    *,
    periods_per_year: float,
    long_only: bool = True,
    initial_equity: float = 1.0,
    start: int = 0,
    rebalance_band: float = 0.02,
) -> BacktestResult:
    """Backtest ``strategy`` on ``bars``: signal, then sizer, then `simulate`.

    Args:
        bars: canonical OHLCV frame. Must not extend beyond the period being evaluated.
        strategy: the signal generator.
        sizer: converts signals to target weights.
        costs: the cost model.
        periods_per_year: for the sizer's vol annualisation (see ``schema.periods_per_year``).
        long_only: clip negative weights to 0. This is required for spot trading.
        initial_equity: starting capital.
        start: first bar index that is traded and scored. Earlier bars are only
            indicator warm-up (used by walk-forward so test folds have history).
        rebalance_band: skip trades that change the weight by this much or less, except
            exits to flat, which always execute.
    """
    validate_schema(bars)
    sig = strategy.signal(bars)
    if not sig.index.equals(bars.index):
        raise ValueError(f"{strategy!r}.signal() must return a Series on bars.index")
    sig = sig.clip(-1.0, 1.0)
    target = sizer.target_weight(sig, bars, periods_per_year)
    result = simulate(
        bars,
        target,
        costs,
        long_only=long_only,
        initial_equity=initial_equity,
        start=start,
        rebalance_band=rebalance_band,
        periods_per_year=periods_per_year,
    )
    result.config.update({"strategy": repr(strategy), "sizer": repr(sizer)})
    return result


def simulate(
    bars: pd.DataFrame,
    target_weights: pd.Series,
    costs: CostModel,
    *,
    periods_per_year: float,
    long_only: bool = True,
    initial_equity: float = 1.0,
    start: int = 0,
    rebalance_band: float = 0.02,
) -> BacktestResult:
    """Simulate trading towards ``target_weights``.

    The weight decided at close[i] fills at open[i+1].

    NaN targets mean flat. The result covers bars ``start..end``. Before ``start``,
    nothing is held or traded.
    """
    validate_schema(bars)
    if not target_weights.index.equals(bars.index):
        raise ValueError("target_weights must be indexed like bars")
    if not 0 <= start < len(bars):
        raise ValueError("start out of range")

    tgt = target_weights.fillna(0.0).to_numpy(dtype=float).copy()
    if long_only:
        tgt = np.clip(tgt, 0.0, None)
    # Nothing is decided before bar start-1. The decision at close[start-1] may fill at
    # open[start], because it only uses data from before the scored period.
    tgt[: max(start - 1, 0)] = 0.0

    o = bars["open"].to_numpy()
    c = bars["close"].to_numpy()
    bar_vol = costs.trailing_bar_vol(bars).to_numpy()
    fee_rate = costs.fee_bps * BP
    slip_rate = costs.slippage_bps * BP + costs.vol_slippage_k * bar_vol
    bar_length = pd.Timedelta(bars.index[1] - bars.index[0]) if len(bars) > 1 else pd.Timedelta(0)
    funding_rate = costs.funding_rate_per_bar(bar_length)

    n = len(bars)
    equity = np.full(n, np.nan)
    held_w = np.zeros(n)
    fees = np.zeros(n)
    slip = np.zeros(n)
    funding = np.zeros(n)
    trades: list[dict[str, Any]] = []

    cash, units = float(initial_equity), 0.0
    for i in range(start, n):
        # 1. open: trade to the target decided at the previous close
        if i > 0:
            eq_open = cash + units * o[i]
            w_before = units * o[i] / eq_open if eq_open > 0 else 0.0
            w_target = tgt[i - 1]
            exit_to_flat = w_target == 0.0 and units != 0.0
            if eq_open > 0 and (abs(w_target - w_before) > rebalance_band or exit_to_flat):
                new_units = w_target * eq_open / o[i]
                qty = new_units - units
                if qty != 0.0:
                    notional = abs(qty) * o[i]
                    fee = notional * fee_rate
                    sl = notional * slip_rate[i]
                    cash -= qty * o[i] + fee + sl
                    units = new_units
                    fees[i], slip[i] = fee, sl
                    trades.append(
                        {
                            "timestamp": bars.index[i],
                            "qty": qty,
                            "price": o[i],
                            "notional": notional,
                            "fee": fee,
                            "slippage": sl,
                            "weight_before": w_before,
                            "weight_after": units * o[i] / (cash + units * o[i]),
                        }
                    )
        # 2. hold: funding on the notional held through the bar (perps only)
        if funding_rate:
            funding[i] = units * o[i] * funding_rate
            cash -= funding[i]
        # 3. close: mark to market
        equity[i] = cash + units * c[i]
        held_w[i] = units * o[i] / (cash + units * o[i]) if (cash + units * o[i]) > 0 else 0.0

    idx = bars.index[start:]
    eq = pd.Series(equity[start:], index=idx, name="equity")
    prev = np.concatenate([[initial_equity], eq.to_numpy()[:-1]])
    rets = pd.Series(eq.to_numpy() / prev - 1.0, index=idx, name="returns")
    trade_cols = ["timestamp", "qty", "price", "notional", "fee", "slippage",
                  "weight_before", "weight_after"]  # fmt: skip
    return BacktestResult(
        equity=eq,
        returns=rets,
        weights=pd.Series(held_w[start:], index=idx, name="weight"),
        target_weights=pd.Series(tgt[start:], index=idx, name="target_weight"),
        trades=pd.DataFrame(trades, columns=trade_cols),
        costs=pd.DataFrame(
            {"fees": fees[start:], "slippage": slip[start:], "funding": funding[start:]},
            index=idx,
        ),
        periods_per_year=periods_per_year,
        config={
            "costs": costs,
            "long_only": long_only,
            "initial_equity": initial_equity,
            "rebalance_band": rebalance_band,
        },
    )
