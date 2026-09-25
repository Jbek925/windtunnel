"""Command-line entry point (``uv run windtunnel ...``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


def _load_bars(args: argparse.Namespace) -> tuple[pd.DataFrame, str, Calendar, list[str]]:
    """Return (bars, title, calendar, notes) from the cache or a synthetic generator."""
    if args.synthetic:
        from windtunnel.data.synthetic import gbm

        bars = gbm(
            args.synthetic,
            mu_ann=args.synthetic_drift,
            sigma_ann=0.6,
            timeframe=args.timeframe,
            seed=args.seed,
        )
        note = (
            "SYNTHETIC random-walk data (no real market). Any 'edge' shown is luck "
            "or a bug; this mode exists to demonstrate the pipeline."
        )
        return bars, f"synthetic GBM {args.timeframe}", "24/7", [note]
    cache = ParquetCache(args.cache)
    source = _source_name(args.source, args.exchange)
    entry = cache.load(source, args.symbol, args.timeframe)
    if entry is None:
        raise SystemExit(
            f"nothing cached for {source}:{args.symbol} {args.timeframe}; "
            "run `windtunnel fetch` first"
        )
    calendar: Calendar = "24/7" if args.source == "ccxt" else "exchange"
    report = validate_bars(
        entry.bars, symbol=args.symbol, timeframe=args.timeframe, calendar=calendar
    )
    notes = list(report.notes)
    if report.has_warnings:
        notes.insert(
            0,
            "Data validation flagged issues. Read them before trusting results:\n"
            + report.summary(max_items=3),
        )
    return entry.bars, f"{args.symbol} {args.timeframe} ({source})", calendar, notes


def _cmd_backtest(args: argparse.Namespace) -> int:
    from windtunnel.backtest.costs import CostModel
    from windtunnel.backtest.sizing import FixedFraction, Sizer, VolTarget
    from windtunnel.data.schema import periods_per_year
    from windtunnel.evaluation.evaluate import evaluate
    from windtunnel.evaluation.report import format_metrics, write_report
    from windtunnel.strategies import STRATEGIES

    if not args.synthetic and (args.source is None or args.symbol is None):
        print("error: give --source and --symbol, or --synthetic N", file=sys.stderr)
        return 2
    bars, title, calendar, notes = _load_bars(args)
    ppy = periods_per_year(args.timeframe, calendar)
    bars_per_day = ppy / 365.0
    costs = CostModel(
        taker_fee_bps=args.fee_bps,
        maker_fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        vol_slippage_k=args.vol_slippage_k,
    )
    sizer: Sizer = (
        VolTarget(target_ann_vol=args.vol_target) if args.sizer == "vol" else FixedFraction(1.0)
    )
    strategy_cls = STRATEGIES[args.strategy]
    notes.append(
        "Long-only (shorts become cash), matching what a spot account can do."
        if not args.long_short
        else "Long/short enabled: shorting is NOT possible in the spot/live setup."
    )
    ev = evaluate(
        bars,
        strategy_cls,
        title=title,
        periods_per_year=ppy,
        costs=costs,
        sizer=sizer,
        train_bars=int(args.train_days * bars_per_day),
        test_bars=int(args.test_days * bars_per_day),
        long_only=not args.long_short,
        prior_trials=args.prior_trials,
        seed=args.seed,
        notes=notes,
    )
    out_dir = (
        Path(args.out)
        / f"{title.split(' (')[0].replace('/', '_').replace(' ', '_')}_{args.strategy}"
    )
    html_path, md_path = write_report(ev, out_dir)
    print(f"== {title}: {args.strategy} (walk-forward, out-of-sample, after costs)\n")
    for line in ev.verdict:
        print(f"* {line}")
    print()
    print(format_metrics([ev.strategy_metrics, *ev.benchmark_metrics]).to_string())
    print(f"\nreport: {html_path}\n        {md_path}")
    return 0


def _make_trader(args: argparse.Namespace) -> Any:
    """Build (runner, store, cfg) from a config file, honouring the live gate."""
    from windtunnel.config import load_config
    from windtunnel.live.gate import resolve_mode
    from windtunnel.paper.broker import Broker, SimulatedBroker
    from windtunnel.paper.feed import CcxtFeed
    from windtunnel.paper.runner import Runner, build_costs, setup_logging
    from windtunnel.paper.store import Store

    cfg = load_config(args.config)
    setup_logging(cfg.log_path)
    gate = resolve_mode(cfg.mode, getattr(args, "live", False))
    store = Store(cfg.db_path)
    if gate.missing:
        store.event(
            "gate",
            "LIVE NOT ENABLED, running as paper. Missing: " + "; ".join(gate.missing),
            "WARNING",
        )
        print("live trading NOT enabled; running as PAPER. Missing: " + "; ".join(gate.missing))
    if store.get("mode") not in (None, gate.mode):
        raise SystemExit(
            f"{cfg.db_path} was used in {store.get('mode')!r} mode; use a separate db_path for "
            f"{gate.mode!r} so paper and real records never mix"
        )
    store.set("mode", gate.mode)
    feed = CcxtFeed(cfg.exchange, cfg.symbol, cfg.timeframe)
    broker: Broker
    if gate.mode == "paper":
        broker = SimulatedBroker(
            store,
            build_costs(cfg),
            strategy_name=cfg.strategy,
            initial_equity=cfg.initial_equity,
            rebalance_band=cfg.rebalance_band,
        )
    else:
        from windtunnel.live.ccxt_broker import LiveCcxtBroker
        from windtunnel.live.gate import load_credentials, make_exchange

        exchange = make_exchange(cfg.exchange, load_credentials())
        live_broker = LiveCcxtBroker(
            exchange,
            cfg.symbol,
            store,
            build_costs(cfg),
            cfg.live,
            strategy_name=cfg.strategy,
            rebalance_band=cfg.rebalance_band,
            shadow=gate.mode == "shadow",
        )
        live_broker.reconcile()
        broker = live_broker
        print(
            f"*** {gate.mode.upper()} MODE: cap {cfg.live.max_live_notional:g} "
            f"{cfg.symbol.split('/')[1]} ***"
        )
    return Runner(cfg, feed, broker, store), store, cfg


def _cmd_paper(args: argparse.Namespace) -> int:
    from windtunnel.paper.runner import compare_with_backtest, run_forever

    if args.paper_cmd == "status":
        from windtunnel.config import load_config
        from windtunnel.paper.store import Store

        cfg = load_config(args.config)
        store = Store(cfg.db_path)
        eq, fills, events = store.table("equity"), store.table("fills"), store.table("events")
        print(
            f"mode: {store.get('mode')}  last bar: {store.get('last_bar')}  "
            f"kill switch: {bool(store.get('kill_switch', False))}"
        )
        if len(eq):
            eq = eq.sort_values("bar_ts")
            first, last = eq.iloc[0], eq.iloc[-1]
            print(
                f"equity {last['equity']:.2f} (start {first['equity']:.2f}), "
                f"position {last['units']:g} units, target {last['target_weight']:.2f}"
            )
        print(f"fills: {len(fills)}")
        print(events.tail(args.events)[["ts", "level", "kind", "message"]].to_string(index=False))
        return 0

    runner, store, cfg = _make_trader(args)
    if args.paper_cmd == "compare":
        bars = runner.feed.snapshot(cfg.history_bars).completed
        both = compare_with_backtest(store, bars, cfg)
        print(both.tail(20).to_string())
        print(f"\nmax |gap| = {both['gap'].abs().max():.5f} (paper vs backtest, rebased to 1.0)")
        return 0
    if args.reset_kill_switch:
        runner.risk.reset_kill_switch()
        print("kill switch reset")
    if args.once:
        res = runner.step()
        print(f"{res.status} bar={res.bar_ts} target={res.target} equity={res.equity}")
        return 0
    run_forever(runner, cfg.poll_seconds)
    return 0


def _cmd_digest(args: argparse.Namespace) -> int:
    from windtunnel.config import load_config
    from windtunnel.paper.digest import build_digest, send_telegram
    from windtunnel.paper.store import Store

    cfg = load_config(args.config)
    text = build_digest(Store(cfg.db_path), cfg, days=args.days)
    print(text)
    if args.dry_run:
        return 0
    try:
        send_telegram(text)
    except RuntimeError as exc:
        print(f"\nnot sent: {exc}", file=sys.stderr)
        return 1
    print("\nsent to Telegram ✅")
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

    from windtunnel.strategies import STRATEGIES

    b = sub.add_parser("backtest", help="walk-forward backtest + honest HTML report")
    add_source_args(b, required=False)
    b.add_argument("--synthetic", type=int, metavar="N", help="use N bars of random-walk data")
    b.add_argument(
        "--synthetic-drift", type=float, default=0.0, help="annual drift for --synthetic"
    )
    b.add_argument("--strategy", choices=sorted(STRATEGIES), default="ma_trend")
    b.add_argument(
        "--sizer",
        choices=["fixed", "vol"],
        default="fixed",
        help="fixed = 100%% per unit signal; vol = volatility targeting",
    )
    b.add_argument("--vol-target", type=float, default=0.2, help="annual vol for --sizer vol")
    b.add_argument("--fee-bps", type=float, default=10.0, help="fee per trade, bps (10 = 0.1%%)")
    b.add_argument("--slippage-bps", type=float, default=2.0)
    b.add_argument("--vol-slippage-k", type=float, default=0.0)
    b.add_argument("--train-days", type=float, default=730, help="walk-forward train window")
    b.add_argument("--test-days", type=float, default=180, help="walk-forward test window")
    b.add_argument("--long-short", action="store_true", help="allow shorts (research only)")
    b.add_argument(
        "--prior-trials",
        type=int,
        default=0,
        help="configs you already tried on this data (be honest; feeds DSR)",
    )
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="reports", help="output directory")
    b.set_defaults(func=_cmd_backtest)

    p = sub.add_parser("paper", help="paper trader (and gated shadow/live trading)")
    psub = p.add_subparsers(dest="paper_cmd", required=True)
    r = psub.add_parser("run", help="run the trading loop")
    r.add_argument("--config", required=True)
    r.add_argument("--once", action="store_true", help="process at most one bar, then exit")
    r.add_argument(
        "--live",
        action="store_true",
        help="one of THREE required opt-ins for real orders (see README)",
    )
    r.add_argument("--reset-kill-switch", action="store_true")
    st = psub.add_parser("status", help="show equity, position and recent events")
    st.add_argument("--config", required=True)
    st.add_argument("--events", type=int, default=15)
    c = psub.add_parser("compare", help="compare paper results with the backtest")
    c.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_paper)

    d = sub.add_parser("digest", help="plain-English weekly summary, sent to Telegram")
    d.add_argument("--config", required=True)
    d.add_argument("--days", type=int, default=7)
    d.add_argument("--dry-run", action="store_true", help="print it, don't send")
    d.set_defaults(func=_cmd_digest)
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
