"""Data-quality checks. Problems are *reported*, never silently fixed.

The report covers:

* **duplicates**: repeated timestamps. The schema forbids these, but raw downloads can
  contain them, so we count them before de-duplication.
* **gaps**: missing bars. For 24/7 data, any step larger than one timeframe. For
  exchange data, only runs longer than a normal weekend plus holiday (> 4 calendar
  days). Missing business days are counted separately, because about 9 NYSE holidays
  a year show up there legitimately.
* **outliers**: bars whose close-to-close log return is more than ``k`` robust standard
  deviations from a trailing median. Crypto has genuinely fat tails, so a flag means
  "look at this bar", not "this bar is wrong".
* **OHLC inconsistencies**: high below max(open, close), or low above min(open, close).
* **zero-volume bars**.
* **survivorship notes**: why results on these hand-picked symbols do not generalise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from windtunnel.data.schema import Calendar, timeframe_to_timedelta

_SURVIVORSHIP: dict[str, str] = {
    "24/7": (
        "BTC and ETH are the two largest survivors of thousands of coins, many of which "
        "went to zero. They were chosen with hindsight, so results on them overstate what "
        "a trend or reversion rule would have earned on 'crypto' chosen ex ante. Exchange "
        "history also only covers pairs that are still listed."
    ),
    "exchange": (
        "SPY, QQQ, GLD and TLT are large, liquid ETFs that exist today; they were chosen "
        "with hindsight. The period covered (mostly post-2004) includes a secular bond "
        "bull market (TLT) and US equity outperformance. Results do not generalise to "
        "'ETFs' or to other decades."
    ),
}


@dataclass
class Gap:
    """A run of missing bars between two observed timestamps."""

    after: str
    before: str
    missing_bars: int


@dataclass
class Outlier:
    """A bar with an unusually large close-to-close log return."""

    timestamp: str
    log_return: float
    robust_z: float


@dataclass
class ValidationReport:
    """The result of `validate_bars`. Serialisable with `to_dict`."""

    symbol: str
    timeframe: str
    calendar: str
    n_bars: int
    start: str | None
    end: str | None
    duplicates: int = 0
    gaps: list[Gap] = field(default_factory=list)
    missing_bars: int = 0
    missing_business_days: int = 0
    outliers: list[Outlier] = field(default_factory=list)
    ohlc_inconsistent: int = 0
    zero_volume: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        """Return True if anything was flagged that a human should look at."""
        return bool(
            self.duplicates
            or self.gaps
            or self.outliers
            or self.ohlc_inconsistent
            or self.zero_volume
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return asdict(self)

    def summary(self, max_items: int = 5) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"{self.symbol} {self.timeframe} ({self.calendar}): {self.n_bars} bars, "
            f"{self.start} → {self.end}",
            f"  duplicates:          {self.duplicates}",
            f"  gaps:                {len(self.gaps)} (missing bars: {self.missing_bars})",
        ]
        if self.calendar == "exchange":
            lines.append(
                f"  missing bus. days:   {self.missing_business_days} "
                "(includes exchange holidays; ~9/yr is normal)"
            )
        lines += [
            f"    {g.after} → {g.before}: {g.missing_bars} bars" for g in self.gaps[:max_items]
        ]
        lines.append(f"  outliers:            {len(self.outliers)}")
        lines += [
            f"    {o.timestamp}: log-ret {o.log_return:+.3f} (robust z {o.robust_z:.1f})"
            for o in self.outliers[:max_items]
        ]
        lines.append(f"  OHLC inconsistent:   {self.ohlc_inconsistent}")
        lines.append(f"  zero-volume bars:    {self.zero_volume}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def validate_bars(
    bars: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    calendar: Calendar,
    outlier_k: float = 10.0,
    outlier_window: int = 100,
) -> ValidationReport:
    """Run the data-quality checks on ``bars`` and return a `ValidationReport`.

    ``bars`` may contain duplicates or be unsorted (raw downloads). Duplicates are
    counted and the checks then run on the sorted, de-duplicated view. The input is
    never modified.
    """
    idx = pd.DatetimeIndex(bars.index)
    report = ValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        calendar=calendar,
        n_bars=len(bars),
        start=str(idx.min()) if len(idx) else None,
        end=str(idx.max()) if len(idx) else None,
        duplicates=int(idx.duplicated().sum()),
    )
    report.notes.append(_SURVIVORSHIP[calendar])
    if len(bars) < 2:
        return report

    df = bars.loc[~idx.duplicated(keep="last")].sort_index()
    ts = pd.DatetimeIndex(df.index)
    td = timeframe_to_timedelta(timeframe)

    steps = ts[1:] - ts[:-1]
    threshold = td if calendar == "24/7" else pd.Timedelta(days=4)
    for i in np.flatnonzero(steps > threshold).tolist():
        after, before = ts[i], ts[i + 1]
        if calendar == "24/7":
            n_missing = int((before - after) / td) - 1
        else:
            n_missing = len(pd.bdate_range(after, before)) - 2
        report.gaps.append(Gap(str(after), str(before), n_missing))
        report.missing_bars += n_missing
    if calendar == "exchange":
        expected = pd.bdate_range(ts[0], ts[-1])
        report.missing_business_days = len(expected.difference(ts))

    logret = pd.Series(np.log(df["close"].to_numpy()), index=ts).diff()
    med = logret.rolling(outlier_window, min_periods=20).median().shift(1)
    mad = (logret - med).abs().rolling(outlier_window, min_periods=20).median().shift(1)
    robust_z = ((logret - med).abs() / (1.4826 * mad)).to_numpy()
    for i in np.flatnonzero(robust_z > outlier_k).tolist():
        report.outliers.append(Outlier(str(ts[i]), float(logret.iloc[i]), float(robust_z[i])))

    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    tol = 1e-9
    bad = (h < np.maximum(o, c) * (1 - tol)) | (lo > np.minimum(o, c) * (1 + tol))
    report.ohlc_inconsistent = int(bad.sum())
    report.zero_volume = int((df["volume"] == 0).sum())
    return report
