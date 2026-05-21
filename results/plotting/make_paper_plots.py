#!/usr/bin/env python3
"""Create paper plots from saved metrics files or W&B summaries.

Inputs can be:
- aggregate JSON files like private/Metrics/Toy/Corrected Reevaluation/*.json;
- CSV exports with metric columns;
- W&B project summaries via --wandb-project ENTITY/PROJECT.

The script writes PNG and PDF files and a normalized CSV used for plotting.
All figures are sized and styled for a 10 pt two-column workshop paper (RLC 2026).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

tmp_dir = Path(os.getenv("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_dir / "xdg-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats


# ── 95% CI helper (t-distribution, for use with seaborn errorbar=) ────────────

def _t_ci95(x: pd.Series):
    """Seaborn-compatible errorbar callable: returns (low_bound, high_bound)."""
    arr = np.array(x.dropna(), dtype=float)
    n = len(arr)
    mean = arr.mean()
    if n < 2:
        return mean, mean
    ci = scipy_stats.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n)
    return mean - ci, mean + ci


def _ci95_scalar(values) -> float:
    """Return the half-width of the 95% CI for a list/array of values."""
    arr = np.array([v for v in values if v is not None and not np.isnan(float(v))], dtype=float)
    n = len(arr)
    if n < 2:
        return float("nan")
    return float(scipy_stats.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n))


# ── Paper-layout constants (RLC 10 pt, 2-column) ──────────────────────────────
COL_W = 3.25    # single-column width, inches
FULL_W = 6.75   # full-page width (both columns + gutter), inches
FONT_SIZE = 8   # body text, pt

# Paul Tol "bright" colorblind-safe palette
# https://personal.sron.nl/~pault/#sec:qualitative
METHOD_PALETTE: dict[str, str] = {
    "continuous_mse": "#BBBBBB",   # grey      — continuous baseline
    "fsq_ce":         "#4477AA",   # blue      — DC-MPC / FSQ
    "vq_ce":          "#66CCEE",   # cyan      — VQ family
    "ddcl_ce":        "#EE6677",   # red       — DDCL CE (stochastic)
    "ddcl_ce_repair": "#AA3377",   # purple    — DDCL CE (repaired)
    "ddcl_soft_ce":   "#CCBB44",   # yellow    — DDCL Soft CE (ours)
    "ddcl_mse":       "#228833",   # green     — DDCL MSE
    "ddcl_cosine":    "#332288",   # indigo    — DDCL cosine (best)
}

METHOD_MARKERS: dict[str, str] = {
    "continuous_mse": "s",
    "fsq_ce":         "^",
    "vq_ce":          "v",
    "ddcl_ce":        "o",
    "ddcl_ce_repair": "p",
    "ddcl_soft_ce":   "D",
    "ddcl_mse":       "X",
    "ddcl_cosine":    "*",
}


def apply_paper_style() -> None:
    """Apply rcParams suitable for a 10 pt two-column workshop paper."""
    plt.rcParams.update({
        # Resolution & font embedding
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "pdf.fonttype":      42,   # TrueType (not Type 3) — required by most venues
        "ps.fonttype":       42,
        # Font sizes (8 pt body, 7 pt ticks/legend)
        "font.size":         FONT_SIZE,
        "axes.titlesize":    FONT_SIZE,
        "axes.labelsize":    FONT_SIZE,
        "xtick.labelsize":   FONT_SIZE - 1,
        "ytick.labelsize":   FONT_SIZE - 1,
        "legend.fontsize":   FONT_SIZE - 1,
        # Line weights
        "lines.linewidth":   1.2,
        "axes.linewidth":    0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        # Spines — remove top and right for clean look
        "axes.spines.top":   False,
        "axes.spines.right": False,
        # Grid
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.5,
        # Legend
        "legend.frameon":    False,
        "legend.handlelength": 1.2,
    })


# ── Metric name registry ──────────────────────────────────────────────────────

METRIC_ALIASES = {
    "success":          "episodic_success",
    "return":           "episodic_return",
    "precision_0p02":   "episodic_precision_success_0p02",
    "precision_0p04":   "episodic_precision_success_0p04",
    "entropy_bits":     "rate/empirical_entropy_bits_per_transition",
    "allocated_bits":   "rate/allocated_bits_per_transition",
    "usage_percent":    "codebook/usage_percent",
    "code_entropy":     "codebook/per_group_entropy_mean",
    "action_saturation": "control/action_saturation_frac",
    "normalized_return": "normalized_return",
}

METHOD_ORDER = [
    "continuous_mse",
    "fsq_ce",
    "vq_ce",
    "ddcl_ce",
    "ddcl_ce_repair",
    "ddcl_soft_ce",
    "ddcl_mse",
    "ddcl_cosine",
]

METHOD_LABELS = {
    "continuous_mse": "Cont-MSE",
    "fsq_ce":         "FSQ-CE",
    "vq_ce":          "VQ-CE",
    "ddcl_ce":        "DDCL-CE",
    "ddcl_ce_repair": "DDCL-CE†",   # dagger — shown only in ablation tables
    "ddcl_soft_ce":   "DDCL-SCE",
    "ddcl_mse":       "DDCL-MSE",
    "ddcl_cosine":    "DDCL-Cos",
}


# ── Data loading helpers ──────────────────────────────────────────────────────

def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and "_type" not in value:
            flat.update(flatten_dict(value, dotted))
        else:
            flat[dotted] = value
    return flat


def clean_metric_name(name: str) -> str:
    clean = str(name)
    for prefix in ("eval_best/.", "eval/.", "rollout/.", "train/."):
        if clean.startswith(prefix):
            clean = clean.replace("/.", "/", 1)
    for prefix in ("eval_best/", "eval/", "rollout/"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    return clean


def infer_method(name: str, config: dict[str, Any] | None = None) -> str:
    lower = name.lower()
    config = config or {}
    quantizer = str(config.get("agent.quantizer", config.get("quantizer", ""))).lower()
    consistency = str(
        config.get("agent.consistency_loss", config.get("consistency_loss", ""))
    ).lower()

    if "continuous" in lower or quantizer == "none":
        return "continuous_mse"
    if "vq" in lower or quantizer == "vq":
        return "vq_ce" if "mse" not in lower and consistency != "mse" else "vq_mse"
    if "fsq" in lower or "dcmpc" in lower or quantizer == "fsq":
        return "fsq_ce" if "mse" not in lower and consistency != "mse" else "fsq_mse"
    if "best-ce" in lower or "repair" in lower:
        return "ddcl_ce_repair"
    if "cosine" in lower or "jepa" in lower or consistency == "cosine":
        return "ddcl_cosine"
    if "soft-ce" in lower or "soft_ce" in lower or (
        quantizer == "ddcl"
        and str(config.get("agent.ddcl_soft_ce_targets",
                           config.get("ddcl_soft_ce_targets", False))).lower() == "true"
    ):
        return "ddcl_soft_ce"
    if "ddcl-mse" in lower or ("ddcl" in lower and "mse" in lower) or (
        quantizer == "ddcl" and consistency == "mse"
    ):
        return "ddcl_mse"
    if "ddcl" in lower or quantizer == "ddcl":
        return "ddcl_ce"
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


def load_aggregate_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for group_name, group_data in data.items():
        metrics = group_data.get("metrics", {})
        values_by_metric = {}
        max_len = 0
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict) and isinstance(stats.get("values"), list):
                clean = clean_metric_name(metric_name)
                values_by_metric[clean] = stats["values"]
                max_len = max(max_len, len(stats["values"]))
        if max_len == 0:
            continue
        method = infer_method(group_name)
        run_ids = group_data.get("ids", [])
        for idx in range(max_len):
            row = {
                "source":       str(path),
                "group":        group_name,
                "method":       method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed_index":   idx,
                "run_id":       run_ids[idx] if idx < len(run_ids) else None,
            }
            for metric_name, values in values_by_metric.items():
                if idx < len(values):
                    row[metric_name] = values[idx]
            rows.append(row)
    return pd.DataFrame(rows)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={col: clean_metric_name(col) for col in df.columns})
    if "method" not in df.columns:
        name_col = next(
            (c for c in ("group", "name", "run_name", "agent") if c in df.columns), None
        )
        df["method"] = df[name_col].map(infer_method) if name_col else "unknown"
    df["method_label"] = df["method"].map(lambda x: METHOD_LABELS.get(x, x))
    df["source"] = str(path)
    return df


def load_wandb(project: str, filters: str | None, max_runs: int) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    parsed_filters = json.loads(filters) if filters else None
    runs = list(
        api.runs(path=project, filters=parsed_filters, per_page=min(max_runs, 100))
    )[:max_runs]
    rows = []
    for run in runs:
        config = flatten_dict(dict(run.config or {}))
        summary = flatten_dict(dict(run.summary or {}))
        row = {
            "source":  project,
            "run_id":  run.id,
            "name":    run.name,
            "group":   getattr(run, "group", None),
            "state":   getattr(run, "state", None),
            "method":  infer_method(run.name or run.id, config),
        }
        row["method_label"] = METHOD_LABELS.get(row["method"], row["method"])
        for key, value in summary.items():
            clean = clean_metric_name(key)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                row[clean] = numeric
        rows.append(row)
    return pd.DataFrame(rows)


def load_inputs(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for path in args.input:
        if path.suffix.lower() == ".json":
            frames.append(load_aggregate_json(path))
        elif path.suffix.lower() == ".csv":
            frames.append(load_csv(path))
        else:
            raise ValueError(f"Unsupported input file type: {path}")
    if args.wandb_project:
        frames.append(load_wandb(args.wandb_project, args.wandb_filters, args.max_runs))
    if not frames:
        raise ValueError("Provide at least one input file or --wandb-project.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    for metric in METRIC_ALIASES.values():
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    # Sanity check
    expected = METRIC_ALIASES["success"]
    if expected not in df.columns or df[expected].notna().sum() == 0:
        import warnings
        warnings.warn(
            f"Column '{expected}' not found or all-NaN after loading. "
            "Check that metric names in the input files match METRIC_ALIASES."
        )
    return df


# ── Figure helpers ────────────────────────────────────────────────────────────

def savefig(outdir: Path, stem: str) -> None:
    """Save current figure as PNG (300 dpi) and PDF (vector) then close."""
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close()


def ordered_methods(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["method"].dropna().tolist()))
    ordered = [m for m in METHOD_ORDER if m in present]
    ordered.extend(m for m in present if m not in ordered)
    return ordered


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_bar(
    df: pd.DataFrame,
    outdir: Path,
    metric: str,
    ylabel: str,
    stem: str,
    *,
    width: float = FULL_W,
    height: float = 2.4,
) -> None:
    """Bar chart (mean ± 95% CI, t-distribution) with individual-seed strip overlay."""
    if metric not in df.columns or df[metric].notna().sum() == 0:
        return
    methods = ordered_methods(df)
    palette = {m: METHOD_PALETTE.get(m, "#888888") for m in methods}
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(width, height))
    sns.barplot(
        data=df,
        x="method",
        y=metric,
        order=methods,
        hue="method",
        hue_order=methods,
        palette=palette,
        dodge=False,
        errorbar=_t_ci95,
        capsize=0.08,
        err_kws={"linewidth": 0.9},
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="method",
        y=metric,
        order=methods,
        hue="method",
        hue_order=methods,
        palette=palette,
        dodge=False,
        alpha=0.55,
        size=2.5,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    savefig(outdir, stem)


def plot_pareto(
    df: pd.DataFrame,
    outdir: Path,
    y_metric: str,
    ylabel: str,
    stem: str,
) -> None:
    """Pareto scatter: per-seed points + per-method mean ± 95% CI error bars.

    x-axis: empirical entropy bits / transition (fair cross-method metric).
    y-axis: *y_metric* (e.g. success or precision@0.02).
    """
    x_metric = METRIC_ALIASES["entropy_bits"]
    if x_metric not in df.columns or y_metric not in df.columns:
        return
    plot_df = df.dropna(subset=[x_metric, y_metric]).copy()
    if plot_df.empty:
        return
    methods = ordered_methods(plot_df)

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    for method in methods:
        mdf = plot_df[plot_df["method"] == method]
        color  = METHOD_PALETTE.get(method, "#888888")
        marker = METHOD_MARKERS.get(method, "o")
        label  = METHOD_LABELS.get(method, method)

        # Individual seeds — small, translucent
        ax.scatter(
            mdf[x_metric], mdf[y_metric],
            c=color, marker=marker, s=14, alpha=0.35, linewidths=0, zorder=2,
        )
        # Per-method mean ± 95% CI (t-distribution)
        mx = mdf[x_metric].mean()
        my = mdf[y_metric].mean()
        sx = _ci95_scalar(mdf[x_metric].tolist())
        sy = _ci95_scalar(mdf[y_metric].tolist())
        ax.errorbar(
            mx, my,
            xerr=sx if not np.isnan(sx) else None,
            yerr=sy if not np.isnan(sy) else None,
            fmt=marker, color=color, markersize=6, markeredgewidth=0.5,
            elinewidth=0.8, capsize=2, label=label, zorder=3,
        )

    ax.set_xlabel("Empirical entropy bits / transition")
    ax.set_ylabel(ylabel)
    ax.legend(title="", loc="best", borderpad=0.4)
    savefig(outdir, stem)


def plot_main_comparison(df: pd.DataFrame, outdir: Path) -> None:
    """Two-panel figure: success (left) + precision@0.02 (right).

    This is the primary comparison figure for the paper.
    Output: main_comparison.png / .pdf  (full-page width, 2.4 in tall)
    """
    m_success = METRIC_ALIASES["success"]
    m_prec = METRIC_ALIASES["precision_0p02"]
    panels = [
        (m_success, "Success"),
        (m_prec,    "Precision success @ 0.02"),
    ]
    panels = [(m, lbl) for m, lbl in panels if m in df.columns and df[m].notna().any()]
    if not panels:
        return
    methods = ordered_methods(df)
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    palette = {m: METHOD_PALETTE.get(m, "#888888") for m in methods}

    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_W, 2.4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (metric, ylabel) in zip(axes, panels):
        sns.barplot(
            data=df,
            x="method",
            y=metric,
            order=methods,
            hue="method",
            hue_order=methods,
            palette=palette,
            dodge=False,
            errorbar=_t_ci95,
            capsize=0.08,
            err_kws={"linewidth": 0.9},
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="method",
            y=metric,
            order=methods,
            hue="method",
            hue_order=methods,
            palette=palette,
            dodge=False,
            alpha=0.55,
            size=2.5,
            linewidth=0,
            legend=False,
            ax=ax,
        )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, None)

    savefig(outdir, "main_comparison")


def plot_code_diagnostics(df: pd.DataFrame, outdir: Path) -> None:
    x_metric = METRIC_ALIASES["entropy_bits"]
    for metric, ylabel, stem in [
        (METRIC_ALIASES["usage_percent"],   "Active codes (%)",                  "code_usage_by_method"),
        (METRIC_ALIASES["code_entropy"],    "Normalized per-group code entropy", "code_entropy_by_method"),
    ]:
        plot_bar(df, outdir, metric, ylabel, stem, width=COL_W, height=2.2)

    y_metric = METRIC_ALIASES["code_entropy"]
    if x_metric in df.columns and y_metric in df.columns:
        plot_pareto(df, outdir, y_metric, "Normalized per-group code entropy",
                    "code_entropy_vs_realized_rate")


def plot_learning_curves(df: pd.DataFrame, outdir: Path) -> None:
    if "env_step" not in df.columns:
        return
    for metric, ylabel, stem in [
        (METRIC_ALIASES["success"],        "Success",                   "curve_success"),
        (METRIC_ALIASES["precision_0p02"], "Precision success @ 0.02",  "curve_precision_0p02"),
        (METRIC_ALIASES["return"],         "Episode return",            "curve_return"),
    ]:
        if metric not in df.columns:
            continue
        plot_df = df.dropna(subset=["env_step", metric])
        if plot_df.empty:
            continue
        methods = ordered_methods(plot_df)
        palette = {m: METHOD_PALETTE.get(m, "#888888") for m in methods}
        fig, ax = plt.subplots(figsize=(COL_W, 2.2))
        sns.lineplot(
            data=plot_df,
            x="env_step",
            y=metric,
            hue="method",
            hue_order=methods,
            palette=palette,
            errorbar="sd",
            ax=ax,
        )
        handles, lbls = ax.get_legend_handles_labels()
        ax.legend(handles, [METHOD_LABELS.get(l, l) for l in lbls], title="")
        ax.set_xlabel("Environment step")
        ax.set_ylabel(ylabel)
        savefig(outdir, stem)


# ── Table export ──────────────────────────────────────────────────────────────

def _fmt_pm(mean: float, ci95: float, decimals: int = 3) -> str:
    """Format mean ± 95% CI as a LaTeX-ready string. Returns '---' for NaN."""
    if pd.isna(mean):
        return "---"
    fmt = f"{{:.{decimals}f}}"
    ci_str = fmt.format(ci95) if not pd.isna(ci95) else "---"
    return f"${fmt.format(mean)} \\pm {ci_str}$"


def write_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    """Write CSV and LaTeX summary tables (mean ± 95% CI per method).

    The ``bold`` flag marks performance columns where higher = better.
    Rate/usage columns are included for completeness but not bolded.
    """
    # (metric_key, header_label, decimal_places, bold_best)
    paper_metrics = [
        (METRIC_ALIASES["success"],        "Success",        3, True),
        (METRIC_ALIASES["precision_0p02"], "Prec@0.02",      3, True),
        (METRIC_ALIASES["precision_0p04"], "Prec@0.04",      3, True),
        (METRIC_ALIASES["return"],         "Return",         2, True),
        (METRIC_ALIASES["entropy_bits"],   "Entropy (bits)", 1, False),   # informational
        (METRIC_ALIASES["allocated_bits"], "Alloc. bits",    1, False),   # informational
        (METRIC_ALIASES["usage_percent"],  "Code use (\\%)", 1, False),   # informational
    ]
    present = [(m, lbl, d, b) for m, lbl, d, b in paper_metrics if m in df.columns]
    if not present:
        return
    metrics = [m for m, _, _, _ in present]
    # Compute mean and 95% CI (t-distribution) per method
    agg_mean = df.groupby("method", sort=False)[metrics].mean()
    agg_ci95 = df.groupby("method", sort=False)[metrics].apply(
        lambda g: g[metrics].apply(_ci95_scalar)
    )
    methods = ordered_methods(df)

    # ── CSV (machine-readable) ────────────────────────────────────────────────
    agg_mean.to_csv(outdir / "summary_table_mean.csv")
    agg_ci95.to_csv(outdir / "summary_table_ci95.csv")

    # ── LaTeX (camera-ready) ─────────────────────────────────────────────────
    col_headers = ["Method"] + [lbl for _, lbl, _, _ in present]
    n_cols = len(col_headers)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Toy precision-gate results (mean $\pm$ 95\% CI over 5 seeds). "
        r"\textbf{Bold} = best per column (performance metrics only).}",
        r"\label{tab:toy_results}",
        r"\begin{tabular}{l" + "c" * (n_cols - 1) + "}",
        r"\toprule",
        " & ".join(col_headers) + r" \\",
        r"\midrule",
    ]

    # Collect formatted cells
    formatted: dict[str, list[str]] = {}
    col_means: dict[str, list[float]] = {m: [] for m, _, _, _ in present}
    ordered_methods_list = [m for m in methods if m in agg_mean.index]
    for method in ordered_methods_list:
        cells = []
        for m, _, d, _ in present:
            mean_val = agg_mean.loc[method, m] if method in agg_mean.index else float("nan")
            ci_val   = agg_ci95.loc[method, m] if method in agg_ci95.index else float("nan")
            cells.append(_fmt_pm(mean_val, ci_val, d))
            col_means[m].append(mean_val if not pd.isna(mean_val) else float("-inf"))
        formatted[method] = cells

    # Bold best value only for performance columns (bold=True)
    for mi, (m, _, _, do_bold) in enumerate(present):
        if not do_bold:
            continue
        vals = col_means[m]
        if not vals:
            continue
        bi = vals.index(max(vals))
        if 0 <= bi < len(ordered_methods_list):
            best_method = ordered_methods_list[bi]
            orig = formatted[best_method][mi]
            formatted[best_method][mi] = r"\textbf{" + orig + "}"

    for method in ordered_methods_list:
        label = METHOD_LABELS.get(method, method)
        cells = [label] + formatted[method]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex_path = outdir / "summary_table.tex"
    latex_path.write_text("\n".join(lines) + "\n")
    print(f"  → LaTeX table written to {latex_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-quality plots from DDCL-MBRL experiment metrics."
    )
    parser.add_argument(
        "--input", type=Path, nargs="*", default=[],
        help="Saved JSON/CSV metrics files.",
    )
    parser.add_argument(
        "--wandb-project", default=None,
        help="Optional W&B project path ENTITY/PROJECT.",
    )
    parser.add_argument("--wandb-filters", default=None, help="Optional W&B API filters JSON.")
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument("--outdir", type=Path, default=Path("paper_plots"))
    args = parser.parse_args()

    apply_paper_style()          # must come before any plt calls

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args)
    df.to_csv(args.outdir / "plot_data_normalized.csv", index=False)
    write_summary_table(df, args.outdir)

    # ── Main paper figure (2-panel) ──────────────────────────────────────────
    plot_main_comparison(df, args.outdir)

    # ── Individual bar charts ────────────────────────────────────────────────
    plot_bar(df, args.outdir, METRIC_ALIASES["success"],
             "Success", "toy_success_by_method")
    plot_bar(df, args.outdir, METRIC_ALIASES["precision_0p02"],
             "Precision success @ 0.02", "toy_precision_0p02_by_method")
    plot_bar(df, args.outdir, METRIC_ALIASES["precision_0p04"],
             "Precision success @ 0.04", "toy_precision_0p04_by_method")
    plot_bar(df, args.outdir, METRIC_ALIASES["return"],
             "Episode return", "return_by_method")
    plot_bar(df, args.outdir, METRIC_ALIASES["entropy_bits"],
             "Empirical entropy bits / transition", "realized_rate_by_method")

    # ── Pareto figures (empirical entropy vs. performance) ───────────────────
    plot_pareto(df, args.outdir, METRIC_ALIASES["success"],
                "Success", "pareto_success_vs_entropy")
    plot_pareto(df, args.outdir, METRIC_ALIASES["precision_0p02"],
                "Precision success @ 0.02", "pareto_precision_0p02_vs_entropy")
    plot_pareto(df, args.outdir, METRIC_ALIASES["return"],
                "Episode return", "pareto_return_vs_entropy")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    plot_code_diagnostics(df, args.outdir)
    plot_learning_curves(df, args.outdir)

    print(f"Wrote plots and tables to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
