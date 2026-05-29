#!/usr/bin/env python3
"""Generate supplementary figures E2, E5, E6 for the RLC 2026 workshop paper.

Reads pre-computed JSON files from private/Metrics/Toy/Corrected Reevaluation/
and produces paper-ready figures.

  Fig 09 (E2) — Gradient-norm comparison (C5/T5):
    Three-panel bar chart (encoder / Q-network / policy grad norm).
    Expected: DDCL-Cos ≈ FSQ-CE (bounded tanh-STE) < VQ-CE for policy/Q gradients.

  Fig 10 (E5) — Codebook usage comparison (C6/T6a):
    Bar chart of final code usage % across methods.
    DDCL ~16–22%, FSQ/VQ ~100% (full utilization, small codebooks).

  Fig 11 (E6) — Planning-variance probe (C1/T1):
    Two-bar chart: Var_stochastic vs Var_deterministic (×21.5 ratio annotated).

All outputs go to DDCL_MBRL_WM_RLC_Workshop/Figures/ as:
  09_e2_grad_norm.{png,pdf}
  10_e5_codebook_usage.{png,pdf}
  11_e6_planning_variance.{png,pdf}

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/gen_supplementary_figs.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as scipy_stats

tmp_dir = Path(os.getenv("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_dir / "matplotlib"))

# ── Paths ──────────────────────────────────────────────────────────────────────
METRICS_DIR = Path("private/Metrics/Toy/Corrected Reevaluation")
PAPER_FIGS  = Path(__file__).parent.parent.parent / "DDCL_MBRL_WM_RLC_Workshop" / "Figures"

# ── Paper style constants ──────────────────────────────────────────────────────
COL_W  = 3.25   # single-column, inches
FULL_W = 6.75   # full-width, inches
FONT   = 8

# Paul Tol "bright" colorblind-safe palette
METHOD_COLORS = {
    "ddcl_ce":      "#EE6677",   # red
    "ddcl_cosine":  "#332288",   # indigo
    "fsq_ce":       "#4477AA",   # blue
    "vq_ce":        "#66CCEE",   # cyan
}
METHOD_LABELS = {
    "ddcl_ce":      "DDCL-CE",
    "ddcl_cosine":  "DDCL-Cos",
    "fsq_ce":       "FSQ-CE",
    "vq_ce":        "VQ-CE",
}
METHOD_ORDER = ["ddcl_ce", "ddcl_cosine", "fsq_ce", "vq_ce"]


def _setup_style() -> None:
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


def _ci95(values: list[float]) -> float:
    arr = np.array([v for v in values if not np.isnan(float(v))], dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    return float(scipy_stats.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n))


def _find_latest(pattern: str) -> Path:
    files = sorted(glob.glob(str(METRICS_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"No file matching: {METRICS_DIR / pattern}")
    return Path(files[-1])


def _savefig(stem: str) -> None:
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PAPER_FIGS / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_FIGS / f"{stem}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PAPER_FIGS / stem}.png / .pdf")


# ── Fig 09 — E2 Gradient-norm comparison ──────────────────────────────────────

def gen_e2_grad_norm() -> None:
    """Three-panel bar chart of encoder / Q-network / policy grad norms."""
    json_path = _find_latest("e2_grad_norm_*.json")
    print(f"  E2 data: {json_path.name}")
    with open(json_path) as f:
        data = json.load(f)

    components = [
        ("encoder",   "Encoder grad norm"),
        ("Q-network", "Q-network grad norm"),
        ("policy",    "Policy grad norm"),
    ]

    methods = [m for m in METHOD_ORDER if m in data]
    x_pos = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]
    xlabels = [METHOD_LABELS[m] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.4))

    for ax, (comp_key, ylabel) in zip(axes, components):
        means, cis, per_seed = [], [], []
        for m in methods:
            gn = data.get(m, {}).get("grad_norm", {}).get(comp_key, {})
            vals = gn.get("values", [])
            means.append(float(np.mean(vals)) if vals else float("nan"))
            cis.append(_ci95(vals))
            per_seed.append(vals)

        ax.bar(
            x_pos, means,
            color=colors, width=0.55,
            yerr=cis, capsize=3.0,
            error_kw={"linewidth": 0.9, "ecolor": "#333333"},
            zorder=2,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(xlabels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)  # grad norms are non-negative

        # Annotate T5 prediction on policy panel
        if comp_key == "policy":
            ax.set_title("T5: DDCL ≈ FSQ < VQ", fontsize=FONT - 1, pad=2)

    fig.suptitle("E2 — Policy / Q-network / Encoder gradient norms (5 seeds)",
                 fontsize=FONT, y=1.02)
    fig.tight_layout(pad=0.8)
    _savefig("09_e2_grad_norm")


# ── Fig 10 — E5 Codebook usage comparison ─────────────────────────────────────

def gen_e5_codebook_usage() -> None:
    """Two-panel bar chart: code usage % and per-group entropy at training end."""
    json_path = _find_latest("e5_codebook_stability_*.json")
    print(f"  E5 data: {json_path.name}")
    with open(json_path) as f:
        data = json.load(f)

    methods = [m for m in METHOD_ORDER if m in data]
    x_pos = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]
    xlabels = [METHOD_LABELS[m] for m in methods]

    panels = [
        ("usage_percent",   "Code usage (%)"),
        ("entropy_mean",    "Per-group code entropy (bits)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5))

    for ax, (cb_key, ylabel) in zip(axes, panels):
        means, cis, per_seed = [], [], []
        for m in methods:
            cb = data.get(m, {}).get("codebook_stability", {}).get(cb_key, {})
            final_vals = cb.get("final_values", [])
            means.append(cb.get("final_mean", float("nan")))
            cis.append(_ci95(final_vals))
            per_seed.append(final_vals)

        ax.bar(
            x_pos, means,
            color=colors, width=0.55,
            yerr=cis, capsize=3.0,
            error_kw={"linewidth": 0.9, "ecolor": "#333333"},
            zorder=2,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(xlabels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        # Fix axis viewport to meaningful range for each panel
        if cb_key == "usage_percent":
            ax.set_ylim(0, 110)  # usage% ∈ [0, 100]; headroom for 100% ref line
            ax.axhline(100, color="#888888", lw=0.6, ls="--", alpha=0.5)
            ax.set_title("T6a: DDCL implicit grid — no collapse",
                         fontsize=FONT - 1, pad=2)
        else:
            ax.set_ylim(bottom=0)  # entropy in bits; auto-extend upward

    fig.suptitle("E5 — Codebook usage at training end (5 seeds)",
                 fontsize=FONT, y=1.02)
    fig.tight_layout(pad=0.8)
    _savefig("10_e5_codebook_usage")


# ── Fig 11 — E6 Planning variance probe ───────────────────────────────────────

def gen_e6_planning_variance() -> None:
    """Two-bar chart: Var_stochastic vs Var_deterministic with ratio annotation."""
    json_path = _find_latest("e6_planning_variance_*.json")
    print(f"  E6 data: {json_path.name}")
    with open(json_path) as f:
        data = json.load(f)

    var_s   = data["var_stochastic"]
    var_d   = data["var_deterministic"]
    std_s   = data["std_stochastic"]
    std_d   = data["std_deterministic"]
    ratio   = data["var_ratio"]
    k_seqs  = data["k_sequences"]
    k_rep   = data["k_rep"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))

    labels = ["Stochastic\n(ε~U)", "Deterministic\n(ε=0)"]
    means  = [var_s, var_d]
    colors = ["#EE6677", "#332288"]

    ax.bar([0, 1], means, color=colors, width=0.55, zorder=2)
    ax.set_yscale("log")

    # ── Bracket + ratio annotation between the two bars ──────────────────────
    # Set y-limits FIRST so the bracket placement is predictable.
    y_lo = var_d / 3.0      # a bit below the smaller bar
    y_hi = var_s * 8.0      # enough headroom for bracket + ratio text
    ax.set_ylim(y_lo, y_hi)

    y_bracket = var_s * 1.6    # bracket sits above the taller bar
    tick_drop = y_bracket / 1.25
    # short vertical ticks dropping from bracket toward each bar top
    ax.plot([0, 0], [tick_drop, y_bracket], color="#444444", lw=0.8, zorder=4)
    ax.plot([1, 1], [tick_drop, y_bracket], color="#444444", lw=0.8, zorder=4)
    # horizontal bar of the bracket
    ax.plot([0, 1], [y_bracket, y_bracket], color="#444444", lw=0.8, zorder=4)
    # ratio label centred on the bracket
    ax.text(0.5, y_bracket * 1.35, f"{ratio:.1f}× variance",
            ha="center", va="bottom", fontsize=FONT, fontweight="bold",
            color="#222222", zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=FONT - 1)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("Return variance / sequence (log scale)")
    ax.set_title(
        f"E6 — MPPI imagined return variance "
        f"({k_seqs} seqs × {k_rep} reps, DDCL-Cos)",
        fontsize=FONT - 1, pad=4,
    )

    fig.tight_layout(pad=0.8)
    _savefig("11_e6_planning_variance")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    _setup_style()
    print("Generating supplementary figures E2, E5, E6 ...")
    print(f"  Output dir: {PAPER_FIGS}")

    print("\n── Fig 09: E2 Gradient-norm comparison ──")
    try:
        gen_e2_grad_norm()
    except Exception as e:
        print(f"  [ERROR] E2: {e}")

    print("\n── Fig 10: E5 Codebook usage ──")
    try:
        gen_e5_codebook_usage()
    except Exception as e:
        print(f"  [ERROR] E5: {e}")

    print("\n── Fig 11: E6 Planning variance ──")
    try:
        gen_e6_planning_variance()
    except Exception as e:
        print(f"  [ERROR] E6: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
