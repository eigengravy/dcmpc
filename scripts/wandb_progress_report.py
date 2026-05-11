#!/usr/bin/env python3
"""Create a compact W&B progress report for DCMPC experiments.

The report is intentionally text-first: it gives Codex enough structured
evidence to analyse training progress and suggest the next experiment without
requiring manual CSV/image exports from W&B.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PRIMARY_METRICS = (
    "eval/.episodic_return",
    "eval/.episodic_success",
    "rollout/.episodic_return",
    "rollout/.episodic_success",
    "eval/episodic_return",
    "eval/episodic_success",
    "rollout/episodic_return",
    "rollout/episodic_success",
    "episodic_return",
    "episodic_success",
)

DEFAULT_DIAGNOSTIC_METRICS = (
    "train/.enc_loss",
    "train/.tc_loss",
    "train/.reward_loss",
    "train/.q_loss",
    "train/.actor_loss",
    "train/.grad_norm",
    "train/.q_grad_norm",
    "train/.pi_grad_norm",
    "train/.exploration_noise",
    "eval/.comms_bits",
    "eval/.rate/ddcl_loss_bits_per_dim",
    "eval/.rate/ddcl_signed_bits_per_transition",
    "eval/.codebook/usage_percent",
    "eval/.codebook/per_group_entropy_mean",
    "eval/.z-rank-1",
    "train/enc_loss",
    "train/tc_loss",
    "train/reward_loss",
    "train/q_loss",
    "train/actor_loss",
    "train/grad_norm",
    "train/q_grad_norm",
    "train/pi_grad_norm",
    "train/exploration_noise",
    "eval/comms_bits",
    "eval/rate/ddcl_loss_bits_per_dim",
    "eval/rate/ddcl_signed_bits_per_transition",
    "eval/quantizer/codebook_usage",
    "eval/quantizer/perplexity",
    "eval/z/rank",
)

LOWER_IS_BETTER_HINTS = (
    "loss",
    "grad_norm",
    "elapsed_time",
    "rollout_time",
    "train_time",
)

CONFIG_KEYS_OF_INTEREST = (
    "env_name",
    "task_name",
    "seed",
    "agent.quantizer",
    "agent.consistency_loss",
    "agent.latent_dim",
    "agent.n_levels",
    "agent.codebook_size",
    "agent.ddcl_lambda",
    "agent.consistency_coef",
    "agent.reward_coef",
    "agent.batch_size",
    "agent.horizon",
    "agent.nstep",
    "agent.utd_ratio",
    "agent.lr",
    "agent.model_lr",
    "agent.q_lr",
    "agent.pi_lr",
)


@dataclass
class MetricStats:
    count: int
    first: float | None
    final: float | None
    best: float | None
    best_step: float | None
    recent_mean: float | None
    previous_mean: float | None
    recent_delta: float | None
    recent_slope: float | None
    volatility: float | None
    direction: str


@dataclass
class RunReport:
    run_id: str
    name: str
    state: str
    url: str
    group: str | None
    tags: list[str]
    config: dict[str, Any]
    summary: dict[str, Any]
    metrics: dict[str, MetricStats]
    missing_metrics: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch W&B histories and write a Markdown progress report."
    )
    parser.add_argument(
        "project",
        help="W&B project path in the form ENTITY/PROJECT, for example my-team/DCWM.",
    )
    parser.add_argument(
        "--filters",
        default=None,
        help=(
            "Optional W&B filters as JSON. Example: "
            '\'{"group": "toy-precision-gate-default"}\''
        ),
    )
    parser.add_argument(
        "--run",
        dest="run_ids",
        action="append",
        default=[],
        help="Specific run id/name to include. Can be passed multiple times.",
    )
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--order", default="-created_at")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--recent-window", type=int, default=8)
    parser.add_argument(
        "--metric",
        dest="extra_metrics",
        action="append",
        default=[],
        help="Extra metric key to analyse. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wandb_progress_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable JSON report path.",
    )
    return parser.parse_args()


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, dotted))
        else:
            flat[dotted] = value
    return flat


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def metric_direction(metric: str) -> str:
    lower = metric.lower()
    if any(hint in lower for hint in LOWER_IS_BETTER_HINTS):
        return "min"
    return "max"


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def collect_series(history: Any, metric: str) -> tuple[list[float], list[float]]:
    if metric not in history:
        return [], []
    step_key = "env_step" if "env_step" in history else "_step"
    xs: list[float] = []
    ys: list[float] = []
    for row_idx, row in history.iterrows():
        y = as_float(row.get(metric))
        if y is None:
            continue
        x = as_float(row.get(step_key))
        xs.append(float(row_idx if x is None else x))
        ys.append(y)
    return xs, ys


def analyse_metric(
    history: Any,
    metric: str,
    recent_window: int,
) -> MetricStats | None:
    xs, ys = collect_series(history, metric)
    if not ys:
        return None

    direction = metric_direction(metric)
    if direction == "min":
        best_idx = min(range(len(ys)), key=lambda idx: ys[idx])
    else:
        best_idx = max(range(len(ys)), key=lambda idx: ys[idx])

    window = max(2, min(recent_window, len(ys)))
    recent = ys[-window:]
    previous = ys[-2 * window : -window] if len(ys) >= 2 * window else []
    recent_mean = safe_mean(recent)
    previous_mean = safe_mean(previous)
    recent_delta = (
        None
        if recent_mean is None or previous_mean is None
        else recent_mean - previous_mean
    )
    volatility = statistics.pstdev(recent) if len(recent) > 1 else 0.0
    recent_slope = linear_slope(xs[-window:], recent)

    return MetricStats(
        count=len(ys),
        first=ys[0],
        final=ys[-1],
        best=ys[best_idx],
        best_step=xs[best_idx],
        recent_mean=recent_mean,
        previous_mean=previous_mean,
        recent_delta=recent_delta,
        recent_slope=recent_slope,
        volatility=volatility,
        direction=direction,
    )


def public_attrs(obj: Any) -> dict[str, Any]:
    data = getattr(obj, "attrs", {}) or {}
    return dict(data) if isinstance(data, dict) else {}


def selected_config(config: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_dict(config)
    selected = {key: flat[key] for key in CONFIG_KEYS_OF_INTEREST if key in flat}
    for key, value in flat.items():
        if key.startswith("hydra."):
            continue
        if key not in selected and len(selected) < 18:
            selected[key] = value
    return selected


def build_run_reports(
    runs: Iterable[Any],
    metrics: list[str],
    samples: int,
    recent_window: int,
) -> list[RunReport]:
    reports: list[RunReport] = []
    for run in runs:
        try:
            history = run.history(samples=samples, keys=["env_step", *metrics], pandas=True)
            if history is not None and history.empty:
                history = run.history(samples=samples, pandas=True)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not fetch history for {run.id}: {exc}", file=sys.stderr)
            history = None

        analysed: dict[str, MetricStats] = {}
        missing: list[str] = []
        if history is not None:
            for metric in metrics:
                stats = analyse_metric(history, metric, recent_window)
                if stats is None:
                    missing.append(metric)
                else:
                    analysed[metric] = stats
        else:
            missing = metrics

        attrs = public_attrs(run)
        summary = dict(getattr(run, "summary", {}) or {})
        config = dict(getattr(run, "config", {}) or {})
        reports.append(
            RunReport(
                run_id=run.id,
                name=getattr(run, "name", run.id),
                state=attrs.get("state", getattr(run, "state", "")),
                url=getattr(run, "url", ""),
                group=attrs.get("group"),
                tags=list(attrs.get("tags", []) or []),
                config=selected_config(config),
                summary=summary,
                metrics=analysed,
                missing_metrics=missing,
            )
        )
    return reports


def choose_primary_metric(reports: list[RunReport]) -> str | None:
    for metric in DEFAULT_PRIMARY_METRICS:
        if any(metric in report.metrics for report in reports):
            return metric
    return None


def get_score(report: RunReport, metric: str) -> tuple[str, MetricStats] | None:
    stats = report.metrics.get(metric)
    if stats is not None and stats.final is not None:
        return metric, stats
    return None


def get_first_available_score(
    report: RunReport,
    candidates: tuple[str, ...],
) -> tuple[str, MetricStats] | None:
    for metric in candidates:
        stats = report.metrics.get(metric)
        if stats is not None and stats.final is not None:
            return metric, stats
    return None


def rank_reports(reports: list[RunReport]) -> list[tuple[RunReport, str, MetricStats]]:
    primary_metric = choose_primary_metric(reports)
    if primary_metric is None:
        return []

    scored = []
    for report in reports:
        score = get_score(report, primary_metric)
        if score is not None:
            metric, stats = score
            scored.append((report, metric, stats))
    return sorted(
        scored,
        key=lambda item: item[2].best if item[2].best is not None else float("-inf"),
        reverse=True,
    )


def generate_suggestions(reports: list[RunReport]) -> list[str]:
    suggestions: list[str] = []
    ranked = rank_reports(reports)

    if not reports:
        return ["No runs matched the query. Check the project path, filters, or W&B login."]

    if not ranked:
        fallback = [
            get_first_available_score(report, DEFAULT_PRIMARY_METRICS)
            for report in reports
        ]
        fallback = [score for score in fallback if score is not None]
        if fallback:
            return [
                "Primary metrics were present inconsistently across runs. Re-run with --metric for the exact key you want Codex to optimise."
            ]
        return [
            "No primary return/success metric was found. Add --metric for the key you care about, or check whether TorchRL logged prefixes such as eval/ and rollout/."
        ]

    best_report, best_metric, best_stats = ranked[0]
    suggestions.append(
        f"Best run by {best_metric} is {best_report.name} ({best_report.run_id}) with best={fmt(best_stats.best)} at env_step={fmt(best_stats.best_step)}."
    )

    active = [report for report in reports if str(report.state).lower() == "running"]
    if active:
        suggestions.append(
            f"{len(active)} run(s) are still running; compare their recent slope against the best completed run before stopping them."
        )

    for report, metric, stats in ranked[:5]:
        if stats.recent_delta is None or stats.recent_slope is None:
            continue
        good_delta = stats.recent_delta > 0 if stats.direction == "max" else stats.recent_delta < 0
        flat = abs(stats.recent_delta) <= max(1e-8, abs(stats.recent_mean or 0.0) * 0.01)
        if flat:
            suggestions.append(
                f"{report.name} appears plateaued on {metric}: recent mean {fmt(stats.recent_mean)} versus previous {fmt(stats.previous_mean)}."
            )
        elif not good_delta:
            suggestions.append(
                f"{report.name} is moving the wrong way on {metric}: recent delta {fmt(stats.recent_delta)}."
            )

    loss_metrics = (
        "train/enc_loss",
        "train/tc_loss",
        "train/reward_loss",
        "train/q_loss",
        "train/actor_loss",
    )
    for report in reports[:10]:
        for metric in loss_metrics:
            stats = report.metrics.get(metric)
            if stats is None or stats.recent_delta is None:
                continue
            if stats.recent_delta > max(1e-8, abs(stats.previous_mean or 0.0) * 0.1):
                suggestions.append(
                    f"{report.name} has rising {metric} over the recent window ({fmt(stats.previous_mean)} -> {fmt(stats.recent_mean)}); inspect learning rate, reward scale, or world-model horizon."
                )
                break

    rate_metrics = (
        "eval/comms_bits",
        "eval/rate/ddcl_loss_bits_per_dim",
        "eval/rate/ddcl_signed_bits_per_transition",
    )
    for report in reports[:10]:
        for metric in rate_metrics:
            stats = report.metrics.get(metric)
            if stats is None or stats.recent_delta is None:
                continue
            if stats.recent_delta > max(1e-8, abs(stats.previous_mean or 0.0) * 0.15):
                suggestions.append(
                    f"{report.name} is using more communication capacity on {metric}; compare return gains against this rate increase before increasing latent/codebook size further."
                )
                break

    unique_suggestions = []
    for suggestion in suggestions:
        if suggestion not in unique_suggestions:
            unique_suggestions.append(suggestion)
    return unique_suggestions[:12]


def metric_table(reports: list[RunReport], metric: str) -> list[str]:
    rows = [
        "| Run | State | Final | Best | Best env_step | Recent delta | Recent slope | Volatility |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        stats = report.metrics.get(metric)
        if stats is None:
            continue
        rows.append(
            "| "
            + " | ".join(
                [
                    report.name.replace("|", "/"),
                    str(report.state),
                    fmt(stats.final),
                    fmt(stats.best),
                    fmt(stats.best_step),
                    fmt(stats.recent_delta),
                    fmt(stats.recent_slope),
                    fmt(stats.volatility),
                ]
            )
            + " |"
        )
    return rows


def render_markdown(
    project: str,
    reports: list[RunReport],
    metrics: list[str],
    suggestions: list[str],
    samples: int,
    recent_window: int,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# W&B Progress Report: {project}",
        "",
        f"Generated: {generated_at}",
        f"Runs analysed: {len(reports)}",
        f"History samples per run: {samples}",
        f"Recent window: {recent_window} logged points",
        "",
        "## Suggestions",
        "",
    ]
    lines.extend(f"- {suggestion}" for suggestion in suggestions)

    lines.extend(["", "## Run Overview", ""])
    lines.extend(
        [
            "| Run | State | Group | Tags | URL |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for report in reports:
        tags = ", ".join(report.tags[:6])
        url = f"[link]({report.url})" if report.url else "-"
        lines.append(
            f"| {report.name.replace('|', '/')} | {report.state} | {report.group or '-'} | {tags or '-'} | {url} |"
        )

    present_metrics = [metric for metric in metrics if any(metric in r.metrics for r in reports)]
    for metric in present_metrics:
        rows = metric_table(reports, metric)
        if len(rows) <= 2:
            continue
        lines.extend(["", f"## Metric: `{metric}`", ""])
        lines.extend(rows)

    lines.extend(["", "## Config Snapshot", ""])
    for report in reports[:12]:
        lines.append(f"### {report.name}")
        if not report.config:
            lines.append("- No selected config fields found.")
            continue
        for key, value in report.config.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    missing_by_metric = {
        metric: sum(metric in report.missing_metrics for report in reports)
        for metric in metrics
    }
    mostly_missing = [
        metric for metric, count in missing_by_metric.items() if count == len(reports)
    ]
    if mostly_missing:
        lines.extend(["", "## Metrics Not Found", ""])
        lines.extend(f"- `{metric}`" for metric in mostly_missing)

    lines.extend(
        [
            "",
            "## Prompt For Codex",
            "",
            "Use this report to identify which run is currently strongest, whether any active runs should continue, and what single next experiment would most reduce uncertainty. Ground suggestions in the metric tables above.",
            "",
        ]
    )
    return "\n".join(lines)


def to_jsonable(reports: list[RunReport], suggestions: list[str]) -> dict[str, Any]:
    return {
        "suggestions": suggestions,
        "runs": [
            {
                "run_id": report.run_id,
                "name": report.name,
                "state": report.state,
                "url": report.url,
                "group": report.group,
                "tags": report.tags,
                "config": report.config,
                "metrics": {
                    metric: stats.__dict__ for metric, stats in report.metrics.items()
                },
                "missing_metrics": report.missing_metrics,
            }
            for report in reports
        ],
    }


def main() -> int:
    args = parse_args()
    if "/" not in args.project:
        print(
            "project must be in ENTITY/PROJECT form, for example my-team/DCWM",
            file=sys.stderr,
        )
        return 2

    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Install project dependencies first.", file=sys.stderr)
        return 2

    filters = json.loads(args.filters) if args.filters else None
    api = wandb.Api()

    if args.run_ids:
        runs = []
        entity, project_name = args.project.split("/", 1)
        project_runs = list(
            api.runs(
                path=f"{entity}/{project_name}",
                filters=filters,
                order=args.order,
                per_page=min(args.max_runs, 100),
            )
        )
        for wanted in args.run_ids:
            match = next(
                (
                    run
                    for run in project_runs
                    if wanted in {run.id, getattr(run, "name", "")}
                ),
                None,
            )
            if match is None:
                match = api.run(f"{args.project}/{wanted}")
            runs.append(match)
    else:
        runs = list(
            api.runs(
                path=args.project,
                filters=filters,
                order=args.order,
                per_page=min(args.max_runs, 100),
            )
        )[: args.max_runs]

    metrics = list(dict.fromkeys([*DEFAULT_PRIMARY_METRICS, *DEFAULT_DIAGNOSTIC_METRICS, *args.extra_metrics]))
    reports = build_run_reports(
        runs=runs,
        metrics=metrics,
        samples=args.samples,
        recent_window=args.recent_window,
    )
    suggestions = generate_suggestions(reports)
    markdown = render_markdown(
        project=args.project,
        reports=reports,
        metrics=metrics,
        suggestions=suggestions,
        samples=args.samples,
        recent_window=args.recent_window,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(to_jsonable(reports, suggestions), indent=2),
            encoding="utf-8",
        )

    print(f"Wrote {args.output}")
    if args.json_output is not None:
        print(f"Wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
