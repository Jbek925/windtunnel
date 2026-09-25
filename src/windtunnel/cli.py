"""Command-line entry point (``uv run windtunnel ...``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import pandas as pd

from windtunnel import __version__
from windtunnel.data.base import DataSource
from windtunnel.data.cache import DEFAULT_ROOT, ParquetCache
from windtunnel.data.schema import Calendar
from windtunnel.data.validate import validate_bars

# The default research universe from the project brief.
DEFAULT_UNIVERSE: list[tuple[str, str, str]] = [
    ("ccxt", "BTC/USDT", "1d"),
    ("ccxt", "BTC/USDT", "1h"),
    ("ccxt", "ETH/USDT", "1d"),
    ("ccxt", "ETH/USDT", "1h"),
    ("yfinance", "SPY", "1d"),
    ("yfinance", "QQQ", "1d"),
    ("yfinance", "GLD", "1d"),
    ("yfinance", "TLT", "1d"),
]


def _make_source(kind: str, exchange: str) -> DataSource:
    if kind == "ccxt":
        from windtunnel.data.ccxt_source import CcxtSource

        return CcxtSource(exchange)
    if kind == "yfinance":
        from windtunnel.data.yfinance_source import YFinanceSource

        return YFinanceSource()
    raise ValueError(f"unknown source {kind!r}")


def _source_name(kind: str, exchange: str) -> str:
    return f"ccxt_{exchange}" if kind == "ccxt" else kind


def _cmd_fetch(args: argparse.Namespace) -> int:
    cache = ParquetCache(args.cache)
    jobs = DEFAULT_UNIVERSE if args.all else [(args.source, args.symbol, args.timeframe)]
    if not args.all and (args.source is None or args.symbol is None):
        print("error: give --source and --symbol, or --all", file=sys.stderr)
        return 2
    start = pd.Timestamp(args.start, tz="UTC")
    failures = 0
    for kind, symbol, timeframe in jobs:
        print(f"== fetching {kind}:{symbol} {timeframe} from {start.date()}")
        try:
            src = _make_source(kind, args.exchange)
            bars, report = cache.update(src, symbol, timeframe, start)
        except Exception as exc:  # report and carry on with the other symbols
            failures += 1
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"   cached {len(bars)} bars → {cache.path_for(src.name, symbol, timeframe)}")
        print(report.summary())
    return 1 if failures else 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cache = ParquetCache(args.cache)
    source = _source_name(args.source, args.exchange)
    entry = cache.load(source, args.symbol, args.timeframe)
    if entry is None:
        print(f"nothing cached for {source}:{args.symbol} {args.timeframe}", file=sys.stderr)
        return 1
    calendar: Calendar = "24/7" if args.source == "ccxt" else "exchange"
    report = validate_bars(
        entry.bars, symbol=args.symbol, timeframe=args.timeframe, calendar=calendar
    )
    print(report.summary(max_items=args.max_items))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="windtunnel", description="Honest research framework for systematic strategies."
    )
    parser.add_argument("--version", action="version", version=f"windtunnel {__version__}")
    parser.add_argument("--cache", default=str(DEFAULT_ROOT), help="parquet cache directory")
    sub = parser.add_subparsers(dest="command")

    def add_source_args(p: argparse.ArgumentParser, required: bool) -> None:
        p.add_argument("--source", choices=["ccxt", "yfinance"], required=required)
        p.add_argument("--symbol", required=required, help="e.g. BTC/USDT or SPY")
        p.add_argument("--timeframe", default="1d", help="1h or 1d (ETFs: 1d only)")
        p.add_argument(
            "--exchange", default="binance", help="ccxt exchange id (e.g. binance, kraken, okx)"
        )

    f = sub.add_parser("fetch", help="download public OHLCV into the parquet cache")
    add_source_args(f, required=False)
    f.add_argument("--all", action="store_true", help="fetch the whole default universe")
    f.add_argument("--start", default="2017-01-01", help="first date (UTC)")
    f.set_defaults(func=_cmd_fetch)

    v = sub.add_parser("validate", help="print the data-quality report for cached bars")
    add_source_args(v, required=True)
    v.add_argument("--max-items", type=int, default=10)
    v.set_defaults(func=_cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
