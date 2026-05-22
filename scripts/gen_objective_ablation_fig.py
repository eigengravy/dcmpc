#!/usr/bin/env python3
"""Fig 4 — DDCL objective ablation figure (C3/C4).

Two-panel bar chart (mean ± 95% CI t-distribution) + per-seed strip overlay
comparing the four DDCL objective variants on the toy precision-gate task.

Shows: CE < SCE < MSE ≈ Cos for prec@0.02 (5 seeds each, all deterministic).
Supports claims C3 (label noise + geometry explain CE gap) and C4 (cosine best).

Data sources:
  - private/Metrics/Toy/Corrected Reevaluation/wandb_toy_all_methods_5seed_heldout_*.json
    (ddcl_ce, ddcl_mse, ddcl_cosine)
  - private/Metrics/Toy/Corrected Reevaluation/wandb_toy_soft_ce_5seed_heldout_*.json
    (ddcl_soft_ce)

Output:
  private/Plots/paper_figs_<date>/fig4_objective_ablation.{png,pdf}

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/gen_objective_ablation_fig.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

os.chdir(_ROOT)

tmp_dir = Path(os.getenv("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_dir / "matplotlib"))

# ── Constants ─────────────────────────────────────────────────────────────────
METRICS_DIR = Path("private/Metrics/Toy/Corrected Reevaluation")
OUT_BASE    = Path("private/Plots/paper_figs")

# Paul Tol "bright" — same as make_paper_plots.py
PALETTE = {
    "ddcl_ce":       "#EE6677",   # red
    "ddcl_soft_ce":  "#CCBB44",   # yellow
    "ddcl_mse":      "#228833",   # green
    "ddcl_cosine":   "#332288",   # indigo
}
MARKERS = {
    "ddcl_ce":       "o",
    "ddcl_soft_ce":  "D",
    "ddcl_mse":      "X",
    "ddcl_cosine":   "*",
}
LABELS = {
    "ddcl_ce":       "DDCL-CE",
    "ddcl_soft_ce":  "DDCL-SCE",
    "ddcl_mse":      "DDCL-MSE",
    "ddcl_cosine":   "DDCL-Cos",
}
METHOD_ORDER = ["ddcl_ce", "ddcl_soft_ce", "ddcl_mse", "ddcl_cosine"]

COL_W  = 3.25
FULL_W = 6.75
FONT   = 8


def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": FONT, "axes.titlesize": FONT,
        "axes.labelsize": FONT, "xtick.labelsize": FONT - 1,
        "ytick.labelsize": FONT - 1, "legend.fontsize": FONT - 1,
        "lines.linewidth": 1.2, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "legend.frameon": False,
    })


def _ci95_halfwidth(values) -> float:
    arr = np.array([v for v in values if v is not None and not np.isnan(float(v))], dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    return float(scipy_stats.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n))


def _load_metric_values(json_path: Path, method_key: str, metric: str) -> list[float]:
    """Load per-seed values for a metric from an aggregate JSON."""
    with open(json_path) as f:
        data = json.load(f)
    entry = data.get(method_key, {})
    metrics = entry.get("metrics", {})
    return [float(v) for v in metrics.get(metric, {}).get("values", [])]


def _find_latest(glob_pattern: str) -> Path:
    files = sorted(glob.glob(str(METRICS_DIR / glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No file matching: {METRICS_DIR / glob_pattern}")
    return Path(files[-1])


def _load_all_data() -> dict[str, dict[str, list[float]]]:
    """Returns {method: {metric: [per-seed values]}}."""
    all_methods_json = _find_latest("wandb_toy_all_methods_5seed_heldout_*.json")
    soft_ce_json     = _find_latest("wandb_toy_soft_ce_5seed_heldout_*.json")

    print(f"  all-methods: {all_methods_json.name}")
    print(f"  soft-CE:     {soft_ce_json.name}")

    metrics_of_interest = [
        "episodic_success",
        "episodic_precision_success_0p02",
        "episodic_precision_success_0p04",
        "episodic_return",
    ]

    data: dict[str, dict[str, list[float]]] = {}
    for method in ["ddcl_ce", "ddcl_mse", "ddcl_cosine"]:
        data[method] = {}
        for metric in metrics_of_interest:
            vals = _load_metric_values(all_methods_json, method, metric)
            if vals:
                data[method][metric] = vals
    # Soft CE from its own JSON
    data["ddcl_soft_ce"] = {}
    for metric in metrics_of_interest:
        vals = _load_metric_values(soft_ce_json, "ddcl_soft_ce", metric)
        if vals:
            data["ddcl_soft_ce"][metric] = vals

    return data


def plot_objective_ablation(data: dict, outdir: Path) -> None:
    """Two-panel bar+strip figure: success (left) + prec@0.02 (right)."""
    _setup_style()

    metrics_panels = [
        ("episodic_success",               "Success"),
        ("episodic_precision_success_0p02", "Precision success @ 0.02"),
    ]

    methods = [m for m in METHOD_ORDER if m in data]
    x_pos   = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))

    for ax, (metric, ylabel) in zip(axes, metrics_panels):
        means, cis = [], []
        for m in methods:
            vals = data[m].get(metric, [])
            means.append(float(np.mean(vals)) if vals else 0.0)
            cis.append(_ci95_halfwidth(vals))

        bars = ax.bar(
            x_pos, means,
            color=[PALETTE[m] for m in methods],
            width=0.55,
            yerr=cis,
            capsize=3.0,
            error_kw={"linewidth": 0.9, "ecolor": "#444444"},
            zorder=2,
        )

        # Per-seed strip overlay
        rng = np.random.default_rng(seed=0)
        for xi, m in enumerate(methods):
            vals = data[m].get(metric, [])
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                xi + jitter, vals,
                color=PALETTE[m], s=10, alpha=0.65, linewidths=0, zorder=3,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=30, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, None)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout(pad=0.8)

    stem = "fig4_objective_ablation"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outdir / stem}.png / .pdf")


def print_summary(data: dict) -> None:
    print("\n── Objective ablation summary ────────────────────────────────────")
    print(f"  {'Method':<12} {'success':>10}  {'prec@0.02':>12}  {'n':>4}")
    for m in METHOD_ORDER:
        if m not in data:
            continue
        s_vals  = data[m].get("episodic_success", [])
        p_vals  = data[m].get("episodic_precision_success_0p02", [])
        s_mean  = float(np.mean(s_vals)) if s_vals else float("nan")
        p_mean  = float(np.mean(p_vals)) if p_vals else float("nan")
        s_ci    = _ci95_halfwidth(s_vals)
        p_ci    = _ci95_halfwidth(p_vals)
        n       = len(s_vals)
        print(f"  {LABELS[m]:<12} {s_mean:.3f}±{s_ci:.3f}  {p_mean:.3f}±{p_ci:.3f}  {n:>4}")


def main():
    date_str = datetime.now().strftime("%Y%m%d")
    outdir   = OUT_BASE / f"paper_figs_{date_str}"

    print("Loading objective ablation data ...")
    data = _load_all_data()
    print_summary(data)

    print("\nGenerating Fig 4 ...")
    plot_objective_ablation(data, outdir)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
