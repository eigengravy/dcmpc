#!/usr/bin/env python3
"""Fig 5 — Stochasticity ablation figure (C1/T1).

Four-panel grouped bar chart showing how stochasticity in eval (ε=0 vs ε~U)
and in training targets (det vs stoch) affects DDCL-CE and DDCL-Cos performance.

Panel layout (1×4): CE(d,d) | CE(s,d) | CE(d,s) | CE(s,s)
                    + Cos(s,d) shown as a separate overlaid bar for reference.

Key prediction (T1): det-eval is the dominant fix (kills Var_quant in MPPI);
det-targets only cleans training labels. DDCL-Cos(s,d) achieves best prec@0.02
via cosine geometry advantage on top of det-targets.

Data source:
  private/Metrics/Toy/Corrected Reevaluation/wandb_toy_stoch_ablation_5seed_heldout_*.json

Output:
  private/Plots/paper_figs_<date>/fig5_stoch_ablation.{png,pdf}

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/gen_stoch_ablation_fig.py
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

# Stochasticity conditions to display (CE ones in 2×2 order, then Cos)
CONDITIONS = [
    # (json_key,        label_short,     eval_mode, tgt_mode, is_cos)
    ("ddcl_ce_dd",  "CE\n(d,d)",      "det",   "det",   False),
    ("ddcl_ce_sd",  "CE\n(s,d)",      "stoch", "det",   False),
    ("ddcl_ce_ds",  "CE\n(d,s)",      "det",   "stoch", False),
    ("ddcl_ce_ss",  "CE\n(s,s)",      "stoch", "stoch", False),
    ("ddcl_cos_sd", "Cos\n(s,d)",     "stoch", "det",   True),
]

# CE palette (shades from light to dark based on stochasticity level)
COND_PALETTE = {
    "ddcl_ce_dd":  "#332288",   # indigo — (d,d) canonical best CE
    "ddcl_ce_sd":  "#AA3377",   # purple — stoch eval
    "ddcl_ce_ds":  "#EE6677",   # red    — stoch targets
    "ddcl_ce_ss":  "#BBBBBB",   # grey   — both stoch (worst)
    "ddcl_cos_sd": "#228833",   # green  — Cos reference
}

COL_W  = 3.25
FULL_W = 6.75
FONT   = 8

METRICS_PANELS = [
    ("episodic_success",                "Success"),
    ("episodic_precision_success_0p02", "Precision success @ 0.02"),
]


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
    arr = np.array([v for v in values if not np.isnan(float(v))], dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    return float(scipy_stats.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n))


def _find_latest(glob_pattern: str) -> Path:
    files = sorted(glob.glob(str(METRICS_DIR / glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No file matching: {METRICS_DIR / glob_pattern}")
    return Path(files[-1])


def load_data() -> dict[str, dict[str, list[float]]]:
    """Returns {condition_key: {metric: [per-seed values]}}."""
    json_path = _find_latest("wandb_toy_stoch_ablation_5seed_heldout_*.json")
    print(f"  Using: {json_path.name}")

    with open(json_path) as f:
        raw = json.load(f)

    all_metrics = [
        "episodic_success",
        "episodic_precision_success_0p02",
        "episodic_precision_success_0p04",
        "episodic_return",
    ]

    data: dict[str, dict[str, list[float]]] = {}
    for key, _, _, _, _ in CONDITIONS:
        entry = raw.get(key, {})
        metrics = entry.get("metrics", {})
        data[key] = {}
        for metric in all_metrics:
            vals = metrics.get(metric, {}).get("values", [])
            if vals:
                data[key][metric] = [float(v) for v in vals]

    return data


def plot_stoch_ablation(data: dict, outdir: Path) -> None:
    """Two-panel bar figure: success (left) + prec@0.02 (right).

    X-axis: 5 conditions: CE(d,d), CE(s,d), CE(d,s), CE(s,s), Cos(s,d)
    Hatched bars for Cos to distinguish from CE conditions.
    """
    _setup_style()

    cond_keys  = [c[0] for c in CONDITIONS]
    cond_labels = [c[1] for c in CONDITIONS]
    is_cos     = [c[4] for c in CONDITIONS]

    x_pos = np.arange(len(cond_keys))

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8))

    for ax, (metric, ylabel) in zip(axes, METRICS_PANELS):
        means, cis = [], []
        for key in cond_keys:
            vals = data.get(key, {}).get(metric, [])
            means.append(float(np.mean(vals)) if vals else 0.0)
            cis.append(_ci95_halfwidth(vals))

        hatches = ["////"] if any(is_cos) else [None] * len(cond_keys)
        hatches = ["////" if ic else "" for ic in is_cos]

        bars = ax.bar(
            x_pos, means,
            color=[COND_PALETTE[k] for k in cond_keys],
            width=0.55,
            yerr=cis,
            capsize=3.0,
            error_kw={"linewidth": 0.9, "ecolor": "#444444"},
            hatch=hatches,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )

        # Per-seed strip overlay
        rng = np.random.default_rng(seed=42)
        for xi, key in enumerate(cond_keys):
            vals = data.get(key, {}).get(metric, [])
            jitter = rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(
                xi + jitter, vals,
                color=COND_PALETTE[key], s=9, alpha=0.65, linewidths=0, zorder=3,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(cond_labels, fontsize=FONT - 1)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, None)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)

        # Annotation: add "(det eval)" / "(stoch eval)" header labels
        ax.axvline(3.5, color="#888888", lw=0.5, ls="--", alpha=0.6, zorder=1)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COND_PALETTE["ddcl_ce_dd"],  label="DDCL-CE (ε=0 eval)"),
        Patch(facecolor=COND_PALETTE["ddcl_ce_sd"],  label="DDCL-CE (ε~U eval)"),
        Patch(facecolor=COND_PALETTE["ddcl_ce_ds"],  label="DDCL-CE (stoch tgt)"),
        Patch(facecolor=COND_PALETTE["ddcl_ce_ss"],  label="DDCL-CE (both stoch)"),
        Patch(facecolor=COND_PALETTE["ddcl_cos_sd"], label="DDCL-Cos (ε~U eval, det tgt)",
              hatch="////", edgecolor="white"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper left",
                   fontsize=FONT - 2, handlelength=1.0)

    fig.tight_layout(pad=0.8)

    stem = "fig5_stoch_ablation"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outdir / stem}.png / .pdf")


def print_summary(data: dict) -> None:
    print("\n── Stochasticity ablation summary ───────────────────────────────")
    print(f"  {'Condition':<16} {'success':>12}  {'prec@0.02':>12}  n")
    for key, label, eval_mode, tgt_mode, is_cos in CONDITIONS:
        s_vals = data.get(key, {}).get("episodic_success", [])
        p_vals = data.get(key, {}).get("episodic_precision_success_0p02", [])
        s_mean = float(np.mean(s_vals)) if s_vals else float("nan")
        p_mean = float(np.mean(p_vals)) if p_vals else float("nan")
        s_ci   = _ci95_halfwidth(s_vals)
        p_ci   = _ci95_halfwidth(p_vals)
        n      = len(s_vals)
        lbl    = label.replace("\n", " ")
        print(f"  {lbl:<16} {s_mean:.3f}±{s_ci:.3f}  {p_mean:.3f}±{p_ci:.3f}  {n}")


def main():
    date_str = datetime.now().strftime("%Y%m%d")
    outdir   = OUT_BASE / f"paper_figs_{date_str}"

    print("Loading stochasticity ablation data ...")
    data = load_data()
    print_summary(data)

    print("\nGenerating Fig 5 ...")
    plot_stoch_ablation(data, outdir)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
