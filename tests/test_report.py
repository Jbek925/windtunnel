from pathlib import Path

import pytest

from tests.helpers import SmaTrend
from windtunnel.backtest.costs import CostModel
from windtunnel.cli import main
from windtunnel.data.synthetic import gbm
from windtunnel.evaluation.evaluate import Evaluation, evaluate
from windtunnel.evaluation.report import format_metrics, to_markdown, write_report


@pytest.fixture(scope="module")
def ev() -> Evaluation:
    return evaluate(
        gbm(1200, seed=8),
        SmaTrend,
        title="test market",
        periods_per_year=365,
        costs=CostModel(),
        train_bars=400,
        test_bars=200,
        n_boot=200,
        notes=["a caveat"],
    )


def test_report_files_written(ev: Evaluation, tmp_path: Path) -> None:
    html_path, md_path = write_report(ev, tmp_path / "r")
    html = html_path.read_text()
    assert "Verdict" in html and "data:image/png;base64," in html
    assert "Deflated Sharpe ratio" in html and "a caveat" in html
    assert "Beat buy &amp; hold? Sharpe advantage" in html
    assert html.count("<img") == 3  # equity, drawdown, sensitivity
    md = md_path.read_text()
    assert md.startswith("# test market: sma_toy")
    assert "| buy & hold" in md or "buy & hold |" in md


def test_every_result_is_shown_with_benchmarks(ev: Evaluation) -> None:
    table = format_metrics([ev.strategy_metrics, *ev.benchmark_metrics])
    assert list(table.columns)[:2] == ["sma_toy", "buy & hold"]
    assert "vol-target" in list(table.columns)[2]
    assert "Sharpe" in table.index and "cost drag (CAGR / yr)" in table.index


def test_benchmarks_cover_the_same_oos_window(ev: Evaluation) -> None:
    oos = ev.wf.oos_returns.index
    for res in ev.benchmark_results.values():
        assert res.returns.index.equals(oos)


def test_markdown_mentions_multiple_testing(ev: Evaluation) -> None:
    assert "Configurations tried" in to_markdown(ev)


def test_cli_backtest_synthetic_keeps_every_run(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    args = [
        "backtest",
        "--synthetic",
        "900",
        "--strategy",
        "ts_momentum",
        "--train-days",
        "300",
        "--test-days",
        "150",
        "--out",
        str(tmp_path),
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "walk-forward, out-of-sample, after costs" in out and "buy & hold" in out
    assert main([*args, "--fee-bps", "40"]) == 0  # a second run must not overwrite the first
    reports = sorted(tmp_path.glob("*/*/report.html"))
    assert len(reports) == 2
    assert any("fee40" in str(p) for p in reports) and any("fee10" in str(p) for p in reports)
