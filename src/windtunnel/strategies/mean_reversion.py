"""Mean reversion: bet that stretched prices snap back towards their recent average.

Economic story: short-term overreaction, liquidity demands and market-maker inventory can
push prices temporarily away from "fair" value. The catch is that in a trending asset,
a price 2σ below its mean is often the start of a crash, not a bargain. Mean reversion
typically has a high win rate with occasional large losses, the reverse of trend following.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from windtunnel.strategies.base import Strategy

EXIT_Z = 0.5
"""Exit threshold, fixed on purpose (not tuned) to keep the parameter count at 2."""


def positions_from_z(z: np.ndarray, entry: float, exit_z: float = EXIT_Z) -> np.ndarray:
    """Run the entry/exit state machine over a z-score path.

    * flat → long when z < -entry, or flat → short when z > +entry
    * long → flat once z >= -exit_z, and short → flat once z <= +exit_z
    * NaN z (warm-up) gives NaN output and resets to flat.

    Each output depends only on z up to that bar, so this is causal.
    """
    if not 0 <= exit_z < entry:
        raise ValueError("need 0 <= exit_z < entry")
    out = np.full(len(z), np.nan)
    pos = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            pos = 0.0
            continue
        if pos == 0.0:
            if zi < -entry:
                pos = 1.0
            elif zi > entry:
                pos = -1.0
        elif (pos > 0 and zi >= -exit_z) or (pos < 0 and zi <= exit_z):
            pos = 0.0
        out[i] = pos
    return out


class ZScoreMeanReversion(Strategy):
    """Trade against large deviations of price from its rolling mean.

    ``z = (close - SMA(lookback)) / std(lookback)``. The rule goes long when
    ``z < -entry``, short when ``z > entry``, and exits when ``|z|`` falls back below 0.5.

    Params (2): ``lookback`` and ``entry``.
    """

    name = "zscore_mr"
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "lookback": [10, 20, 50],
        "entry": [1.0, 1.5, 2.0],
    }

    def zscore(self, bars: pd.DataFrame) -> pd.Series:
        """Return the rolling z-score of close, using only data up to each bar."""
        close = bars["close"]
        n = self.params["lookback"]
        mean = close.rolling(n).mean()
        std = close.rolling(n).std()
        return (close - mean) / std.where(std > 0)

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        """Return +1 (long), -1 (short), 0 (flat), or NaN during warm-up."""
        z = self.zscore(bars).to_numpy(dtype=float)
        return pd.Series(positions_from_z(z, float(self.params["entry"])), index=bars.index)
