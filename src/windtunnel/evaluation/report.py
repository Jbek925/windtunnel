"""Write an `Evaluation` as a self-contained HTML report plus a markdown summary.

The HTML embeds its charts as PNGs, so it's a single file you can open offline or
email. The colours come from a colour-blind-validated palette: strategy blue, buy & hold
orange, vol-target B&H aqua. The heatmap uses a diverging blue↔red scale centred on a
Sharpe of 0, so red means "lost money after costs".
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (servers, CI)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator

from windtunnel.evaluation.evaluate import Evaluation
from windtunnel.evaluation.metrics import Metrics, drawdown, metrics_table, sharpe_ratio

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # strategy, buy & hold, vol-target B&H
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
DIVERGING = LinearSegmentedColormap.from_list("wt_div", ["#e34948", "#f0efec", "#2a78d6"])


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def _png(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _curves(ev: Evaluation) -> dict[str, pd.Series]:
    out = {ev.strategy_name: ev.wf.oos_returns}
    out.update({k: r.returns for k, r in ev.benchmark_results.items()})
    return out


def _spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Nudge label y-positions (in log10 space) apart so end-of-line labels don't collide."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)[order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + min_gap)
    out = np.empty_like(placed)
    out[order] = placed
    return [float(v) for v in out]


def equity_chart(ev: Evaluation) -> str:
    """Return a PNG (base64) of out-of-sample equity, strategy vs benchmarks, on a log scale."""
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    _style(ax)
    curves = [((1 + r).cumprod(), label) for label, r in _curves(ev).items()]
    for (eq, label), color in zip(curves, SERIES, strict=False):
        ax.plot(eq.index, eq.to_numpy(), color=color, linewidth=1.6, label=label)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    lo, hi = np.log10(ax.get_ylim())
    ends = [float(np.log10(eq.iloc[-1])) for eq, _ in curves]
    for (eq, label), y in zip(curves, _spread_labels(ends, (hi - lo) * 0.06), strict=True):
        ax.annotate(
            f" {label}  {eq.iloc[-1]:.2f}×",
            (eq.index[-1], 10**y),
            color=INK_2,
            fontsize=8,
            va="center",
            annotation_clip=False,
        )
    ax.set_ylabel("growth of 1 (log scale)", color=INK_2, fontsize=9)
    ax.set_title("Out-of-sample equity, after costs", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    ax.margins(x=0.02)
    return _png(fig)


def drawdown_chart(ev: Evaluation) -> str:
    """Return a PNG (base64) of drawdowns, strategy vs benchmarks."""
    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=SURFACE)
    _style(ax)
    for (label, rets), color in zip(_curves(ev).items(), SERIES, strict=False):
        dd = drawdown(rets)
        ax.plot(dd.index, dd.to_numpy() * 100, color=color, linewidth=1.2, label=label)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("drawdown (%)", color=INK_2, fontsize=9)
    ax.set_title("Drawdown from previous peak", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    ax.margins(x=0.02)
    return _png(fig)


def sensitivity_chart(ev: Evaluation) -> str | None:
    """Return a PNG (base64) heatmap (2 params) or bar chart (1 param) of full-sample Sharpe."""
    sens = ev.sensitivity
    params = [c for c in sens.columns if c not in ("sharpe", "n_trades")]
    if not params:
        return None
    lim = max(float(sens["sharpe"].abs().max()), 0.1)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    if len(params) == 1:
        fig, ax = plt.subplots(figsize=(5, 2.6), facecolor=SURFACE)
        _style(ax)
        labels = sens[params[0]].astype(str)
        ax.bar(
            labels,
            sens["sharpe"],
            color=DIVERGING(norm(sens["sharpe"].to_numpy())),
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xlabel(params[0], color=INK_2, fontsize=9)
        ax.set_ylabel("Sharpe (after costs)", color=INK_2, fontsize=9)
    else:
        grid = sens.pivot_table(index=params[0], columns=params[1], values="sharpe")
        fig, ax = plt.subplots(figsize=(5, 2.6), facecolor=SURFACE)
        ax.imshow(grid.to_numpy(), cmap=DIVERGING, norm=norm, aspect="auto")
        ax.set_xticks(range(grid.shape[1]), [str(c) for c in grid.columns])
        ax.set_yticks(range(grid.shape[0]), [str(i) for i in grid.index])
        ax.set_xlabel(params[1], color=INK_2, fontsize=9)
        ax.set_ylabel(params[0], color=INK_2, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=9)
        for (i, j), v in np.ndenumerate(grid.to_numpy()):
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#ffffff" if abs(v) > 0.6 * lim else INK,
                )
        for side in ax.spines.values():
            side.set_visible(False)
    ax.set_title(
        "Parameter sensitivity: full-sample Sharpe (IN-SAMPLE)", loc="left", color=INK, fontsize=11
    )
    return _png(fig)


_PCT = {"total_return", "cagr", "ann_vol", "max_drawdown", "cost_drag", "win_rate", "exposure"}
_NAMES = {
    "start": "start",
    "end": "end",
    "years": "years",
    "total_return": "total return",
    "cagr": "CAGR",
    "ann_vol": "annual volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "max drawdown",
    "max_dd_days": "longest drawdown (days)",
    "turnover": "turnover (× equity / yr)",
    "cost_drag": "cost drag (CAGR / yr)",
    "n_trades": "fills",
    "n_round_trips": "round trips",
    "win_rate": "win rate",
    "exposure": "time in market",
}


def format_metrics(metrics: list[Metrics]) -> pd.DataFrame:
    """Return a display table: human-readable row names and formatted values."""
    table = metrics_table(metrics)
    out = table.copy().astype(object)
    for key in table.index:
        for col in table.columns:
            v = table.loc[key, col]
            if isinstance(v, str):
                continue
            fv = float(v)
            if not np.isfinite(fv):
                out.loc[key, col] = "n/a"
            elif key in _PCT:
                out.loc[key, col] = f"{fv:.1%}"
            elif key in ("n_trades", "n_round_trips", "max_dd_days"):
                out.loc[key, col] = f"{fv:,.0f}"
            else:
                out.loc[key, col] = f"{fv:.2f}"
    out.index = [_NAMES.get(str(k), str(k)) for k in out.index]
    return out


def _fold_table(ev: Evaluation) -> pd.DataFrame:
    rows = []
    for k, f in enumerate(ev.wf.folds):
        rows.append(
            {
                "fold": k + 1,
                "train": f"{f.train_start.date()} → {f.train_end.date()}",
                "test": f"{f.test_start.date()} → {f.test_end.date()}",
                "chosen params": ", ".join(f"{a}={b}" for a, b in f.best_params.items()) or "—",
                "train Sharpe": f"{f.train_score:.2f}",
                "test Sharpe": f"{sharpe_ratio(f.test_returns, ev.wf.periods_per_year):.2f}",
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Return a GitHub-flavoured markdown table (avoids pandas' optional tabulate dep)."""
    cols = ([""] if index else []) + [str(c) for c in df.columns]
    rows = [
        ([str(i)] if index else []) + [str(v) for v in row]
        for i, row in zip(df.index, df.to_numpy().tolist(), strict=True)
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _all_metrics(ev: Evaluation) -> list[Metrics]:
    return [ev.strategy_metrics, *ev.benchmark_metrics, ev.strategy_metrics_costs_x2]


def _robustness_rows(ev: Evaluation) -> list[tuple[str, str, str]]:
    ci, dsr = ev.sharpe_ci, ev.dsr
    return [
        (
            "Beat buy & hold? Sharpe advantage, 95% CI",
            f"{ev.sharpe_diff_ci.sharpe:+.2f} [{ev.sharpe_diff_ci.low:+.2f}, "
            f"{ev.sharpe_diff_ci.high:+.2f}]",
            "THE key test. Includes 0 → not proven better than simply holding."
            if not ev.sharpe_diff_ci.excludes_zero
            else "Above 0 → beat holding by more than luck (on this one path).",
        ),
        (
            "Made money at all? Sharpe 95% CI",
            f"[{ci.low:.2f}, {ci.high:.2f}]",
            "Includes 0 → indistinguishable from luck."
            if not ci.excludes_zero
            else "Excludes 0, but holding the asset may have made money too; see the row above.",
        ),
        (
            "Configurations tried (N)",
            f"{dsr.n_trials}",
            "Grid size plus any --prior-trials. Every try inflates the best result.",
        ),
        (
            "Luck threshold (best of N useless)",
            f"{dsr.sr_threshold_annual:.2f}",
            "The Sharpe you'd expect from the luckiest of N skill-less strategies.",
        ),
        (
            "Deflated Sharpe ratio",
            f"{dsr.dsr:.2f}",
            "P(true Sharpe > luck threshold). Want > 0.95. Measures 'made money', not "
            "'beat holding'.",
        ),
        (
            "Prob. Sharpe > 0 (PSR)",
            f"{dsr.psr_vs_zero:.2f}",
            "Ignores multiple testing; always ≥ DSR.",
        ),
        (
            "Sanity: advantage over buy & hold on shuffled data",
            f"{ev.shuffled_edge:+.2f}",
            f"Should be ≈ 0 or negative (shuffling destroys trends). Raw shuffled Sharpe "
            f"{ev.shuffled_sharpe:.2f} includes the asset's drift. A clearly positive "
            "advantage suggests a bug.",
        ),
    ]


def to_markdown(ev: Evaluation) -> str:
    """Return the report as markdown (tables + verdict, no charts)."""
    lines = [f"# {ev.title}: {ev.strategy_name}", "", "## Verdict", ""]
    lines += [f"- {v}" for v in ev.verdict]
    lines += [
        "",
        "## Out-of-sample results (after costs)",
        "",
        _md_table(format_metrics(_all_metrics(ev))),
        "",
    ]
    lines += ["## Robustness", "", "| check | value | meaning |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in _robustness_rows(ev)]
    lines += ["", "## Walk-forward folds", "", _md_table(_fold_table(ev), index=False), ""]
    lines += ["## Notes and caveats", ""] + [f"- {n}" for n in ev.notes]
    lines += ["", "## Configuration", ""] + [f"- **{k}**: {v}" for k, v in ev.config.items()]
    return "\n".join(lines) + "\n"


_CSS = """
:root { --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --line:#e1e0d9; --warn-bg:#fff4e0; --warn-line:#eda100; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d0d0d; --card:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
          --line:#2c2c2a; --warn-bg:#2a2210; --warn-line:#c98500; }
}
body { background:var(--bg); color:var(--ink); margin:0;
       font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; }
.sub { color: var(--ink2); margin: 0 0 16px; }
.verdict { background: var(--warn-bg); border-left: 4px solid var(--warn-line);
           padding: 12px 16px; border-radius: 6px; }
.verdict li { margin: 4px 0; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
        padding: 12px; overflow-x: auto; }
.chart { background:#fcfcfb; border-radius: 8px; padding: 8px; margin: 12px 0; }
.chart img { width: 100%; height: auto; display: block; }
.chart.narrow { max-width: 560px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--line);
         white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
td.wrap { white-space: normal; text-align: left; color: var(--ink2); }
ul.notes { color: var(--ink2); }
"""


def _html_table(df: pd.DataFrame, index: bool = True) -> str:
    return df.to_html(index=index, escape=True, border=0)


def to_html(ev: Evaluation) -> str:
    """Return the full report as one self-contained HTML string."""
    charts = [equity_chart(ev), drawdown_chart(ev)]
    sens = sensitivity_chart(ev)
    rob = "".join(
        f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td>"
        f"<td class='wrap'>{html.escape(c)}</td></tr>"
        for a, b, c in _robustness_rows(ev)
    )
    img = "".join(
        f"<div class='chart'><img alt='chart' src='data:image/png;base64,{c}'></div>"
        for c in charts
    )
    sens_html = (
        f"<div class='chart narrow'><img alt='parameter sensitivity' "
        f"src='data:image/png;base64,{sens}'></div>"
        "<p class='sub'>In-sample over the whole period, so this shows the landscape's "
        "<em>shape</em>, not achievable returns. A lone bright cell among red ones is a "
        "warning sign of overfitting. Broad plateaus are more believable.</p>"
        if sens
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(ev.title)} backtest</title><style>{_CSS}</style></head>
<body><main>
<h1>{html.escape(ev.title)}: {html.escape(ev.strategy_name)}</h1>
<p class="sub">Walk-forward, out-of-sample, after costs. {html.escape(ev.config["bars"])}</p>
<h2>Verdict</h2>
<div class="verdict"><ul>{"".join(f"<li>{html.escape(v)}</li>" for v in ev.verdict)}</ul></div>
<h2>Equity and drawdown</h2>{img}
<h2>Results table (out-of-sample, after costs)</h2>
<div class="card">{_html_table(format_metrics(_all_metrics(ev)))}</div>
<h2>Robustness and honesty checks</h2>
<div class="card"><table><tr><th>check</th><th>value</th><th>meaning</th></tr>{rob}</table></div>
{sens_html}
<h2>Walk-forward folds</h2>
<div class="card">{_html_table(_fold_table(ev), index=False)}</div>
<h2>Notes and caveats</h2>
<ul class="notes">{"".join(f"<li>{html.escape(n)}</li>" for n in ev.notes)}</ul>
<h2>Configuration</h2>
<div class="card"><table>{
        "".join(
            f"<tr><td>{html.escape(str(k))}</td><td class='wrap'>{html.escape(str(v))}</td></tr>"
            for k, v in ev.config.items()
        )
    }</table></div>
</main></body></html>
"""


def write_report(ev: Evaluation, out_dir: Path | str) -> tuple[Path, Path]:
    """Write ``report.html`` and ``report.md`` into ``out_dir`` and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path, md_path = out / "report.html", out / "report.md"
    html_path.write_text(to_html(ev), encoding="utf-8")
    md_path.write_text(to_markdown(ev), encoding="utf-8")
    return html_path, md_path
