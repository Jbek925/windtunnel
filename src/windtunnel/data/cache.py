"""Parquet cache for bar data, with a JSON metadata sidecar and incremental updates.

Layout: ``{root}/{source}/{symbol_sanitised}/{timeframe}.parquet`` and ``.meta.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from windtunnel.data.base import DataSource
from windtunnel.data.schema import timeframe_to_timedelta, to_canonical, validate_schema
from windtunnel.data.validate import ValidationReport, validate_bars

DEFAULT_ROOT = Path("data/cache")


def _sanitise(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", symbol)


@dataclass
class CacheEntry:
    """The bars loaded from the cache, plus their metadata."""

    bars: pd.DataFrame
    meta: dict[str, Any]


class ParquetCache:
    """Read and write canonical bar frames as parquet files."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def path_for(self, source: str, symbol: str, timeframe: str) -> Path:
        """Return the parquet path for a (source, symbol, timeframe) triple."""
        return self.root / source / _sanitise(symbol) / f"{timeframe}.parquet"

    def load(self, source: str, symbol: str, timeframe: str) -> CacheEntry | None:
        """Load cached bars, or return ``None`` if nothing is cached."""
        path = self.path_for(source, symbol, timeframe)
        if not path.exists():
            return None
        bars = to_canonical(pd.read_parquet(path))
        validate_schema(bars)
        meta_path = path.with_suffix(".meta.json")
        meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return CacheEntry(bars, meta)

    def save(
        self, bars: pd.DataFrame, source: str, symbol: str, timeframe: str, meta: dict[str, Any]
    ) -> Path:
        """Validate and write ``bars`` plus metadata. Returns the parquet path."""
        validate_schema(bars)
        path = self.path_for(source, symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        bars.to_parquet(tmp)
        tmp.replace(path)  # atomic on POSIX: never leaves a half-written file
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return path

    def update(
        self,
        source: DataSource,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        now: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Fetch new bars from ``source``, merge them into the cache and re-validate.

        When a cache exists, only bars from the last cached bar onward are fetched (the
        last bar is re-fetched in case it was revised). Where old and new rows overlap,
        the newly fetched values win. The validation report covers the full merged series.
        """
        existing = self.load(source.name, symbol, timeframe)
        fetch_from = start
        if existing is not None and len(existing.bars):
            fetch_from = max(start, pd.Timestamp(existing.bars.index[-1]))
        new = source.fetch(symbol, timeframe, fetch_from)
        revised = 0
        if existing is None:
            merged = new
        else:
            overlap = existing.bars.index.intersection(new.index)
            old_rows = existing.bars.loc[overlap]
            new_rows = new.loc[overlap]
            revised = int((~np.isclose(old_rows, new_rows, rtol=1e-9)).any(axis=1).sum())
            merged = pd.concat([existing.bars.drop(overlap), new]).sort_index()
        merged = merged.loc[merged.index >= start]
        report = validate_bars(merged, symbol=symbol, timeframe=timeframe, calendar=source.calendar)
        if revised:
            report.notes.append(
                f"{revised} previously cached bar(s) were revised by the source and overwritten "
                "(normal for dividend-adjusted ETF prices; suspicious for crypto)."
            )
        meta = {
            "source": source.name,
            "symbol": symbol,
            "timeframe": timeframe,
            "calendar": source.calendar,
            "bar_length": str(timeframe_to_timedelta(timeframe)),
            "index_convention": "bar open time, UTC",
            "fetched_at": str(now or pd.Timestamp.now(tz="UTC")),
            "rows": len(merged),
            "validation": report.to_dict(),
        }
        self.save(merged, source.name, symbol, timeframe, meta)
        return merged, report
