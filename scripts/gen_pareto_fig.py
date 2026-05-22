#!/usr/bin/env python3
"""Fig 3 — Rate-precision Pareto: two views (C2/C7/C9/T6).

TWO-PANEL figure comparing DDCL, FSQ, and VQ at multiple code-budget levels:

Panel A — Matched-capacity headline (controlled budget):
  x = nominal capacity (bits), identical by design across methods at each level.
  y = Precision success @ 0.02 (on-policy, 3 seeds, mean ± 95% CI).
  Budget levels: 256 bits (VQ-16, FSQ-[4,4], DDCL-s1.0) and
                 384 bits (VQ-64, FSQ-[8,8], DDCL-s3.0).
  Each method is plotted as a scatter+CI point. For DDCL, the BEST λ at
  each scale is shown (maximizes prec@0.02). Multiple DDCL λ values are shown
  as a λ-sweep curve to illustrate bias-variance trade-off (C9/T7).

Panel B — Realized-rate panel (measured rate on D_ref):
  x = R_m = code entropy bits/transition on D_ref (shared uniform grid, ε=0).
  y = Precision success @ 0.02.
  Same methods; DDCL points now shift left (self-compression via rate penalty).
  VQ/FSQ stay near their nominal values (only tanh saturation compresses).

The two panels share the same y-axis range for visual alignment.
Vertical dashed lines mark the matched-budget levels (256, 384 bits) on both
panels to link the "controlled rate" and "measured rate" views.

Data sources:
  - private/Metrics/Toy/QuantizerPareto/pareto_sweep_aggregated.json
    (on-policy performance for all Pareto sweep conditions, 3 seeds)
  - private/Metrics/Toy/Corrected Reevaluation/e7_shared_dataset_rate_*.json
    (realized code entropy on D_ref for each method)

Output:
  private/Plots/paper_figs_<date>/fig3_pareto.{png,pdf}

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/gen_pareto_fig.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats as scipy_stats

os.chdir(_ROOT)

tmp_dir = Path(os.getenv("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_dir / "matplotlib"))

# ── File paths ─────────────────────────────────────────────────────────────────
PARETO_JSON = Path("private/Metrics/Toy/QuantizerPareto/pareto_sweep_aggregated.json")
E7_DIR      = Path("private/Metrics/Toy/Corrected Reevaluation")
OUT_BASE    = Path("private/Plots/paper_figs")

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "ddcl": "#332288",   # indigo
    "fsq":  "#4477AA",   # blue
    "vq":   "#66CCEE",   # cyan
}
MARKERS = {
    "ddcl": "*",
    "fsq":  "^",
    "vq":   "v",
}
FULL_W = 6.75
FONT   = 8

# ── Matched-capacity budget levels ─────────────────────────────────────────────
# At each budget, VQ-K, FSQ-[K,K], DDCL-sX all have the SAME nominal bits.
BUDGETS = [
    {
        "bits": 256.0,
        "ddcl_scales": [1.0],  # nominal 256 bits
        "fsq_key":  "fsq_[4, 4]",
        "vq_key":   "vq_16",
        "ddcl_keys": [f"s1_d1_l{l}" for l in ["0.003", "0.001", "0.0003", "0.0001"]],
        "e7_fsq_key":  "pareto_fsq_4x4",
        "e7_vq_key":   "pareto_vq_16",
        "e7_ddcl_keys": {
            "s1_d1_l0.003":  "pareto_ddcl_s1p0_l0.003",
            "s1_d1_l0.001":  "pareto_ddcl_s1p0_l0.001",
            "s1_d1_l0.0003": "pareto_ddcl_s1p0_l0.0003",
            "s1_d1_l0.0001": "pareto_ddcl_s1p0_l0.0001",
        },
    },
    {
        "bits": 384.0,
        "ddcl_scales": [3.0],
        "fsq_key": "fsq_[8, 8]",
        "vq_key":  "vq_64",
        "ddcl_keys": [f"s3_d1_l{l}" for l in ["0.003", "0.001", "0.0003", "0.0001"]],
        "e7_fsq_key": "pareto_fsq_8x8",
        "e7_vq_key":  "pareto_vq_64",
        "e7_ddcl_keys": {
            "s3_d1_l0.003":  "pareto_ddcl_s3p0_l0.003",
            "s3_d1_l0.001":  "pareto_ddcl_s3p0_l0.001",
            "s3_d1_l0.0003": "pareto_ddcl_s3p0_l0.0003",
            "s3_d1_l0.0001": "pareto_ddcl_s3p0_l0.0001",
        },
    },
]

Y_METRIC = "episodic_precision_success_0p02"
Y_LABEL  = "Precision success @ 0.02"


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
        "legend.frameon": False, "legend.handlelength": 1.2,
    })


def _ci95(values) -> float:
    arr = np.array([v for v in values if not np.isnan(float(v))], dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    return float(scipy_stats.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n))


def _find_latest_e7() -> Path:
    files = sorted(glob.glob(str(E7_DIR / "e7_shared_dataset_rate_*.json")))
    if not files:
        raise FileNotFoundError(f"No E7 JSON found in {E7_DIR}")
    return Path(files[-1])


def load_pareto() -> dict:
    with open(PARETO_JSON) as f:
        return json.load(f)


def load_e7() -> dict:
    path = _find_latest_e7()
    print(f"  E7 rate data: {path.name}")
    with open(path) as f:
        return json.load(f)


def get_perf(pareto: dict, method_type: str, key: str) -> tuple[float, float, float]:
    """Returns (mean, ci95, n) for Y_METRIC.

    Works with both:
      - {"values": [...]} (from heldout JSONs)
      - {"mean": x, "std": y, "n": z} (from pareto_sweep_aggregated.json)
    """
    entry = pareto.get(method_type, {}).get(key, {})
    metric_entry = entry.get(Y_METRIC, {})
    if not metric_entry:
        return float("nan"), 0.0, 0

    # Prefer per-seed values for exact CI
    vals = metric_entry.get("values", [])
    if vals:
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), _ci95(vals), len(arr)

    # Fall back to mean/std/n
    mean = metric_entry.get("mean")
    std  = metric_entry.get("std")
    n    = int(metric_entry.get("n", 0))
    if mean is None or n == 0:
        return float("nan"), 0.0, 0
    if std is None or n < 2:
        return float(mean), 0.0, n
    ci = float(scipy_stats.t.ppf(0.975, df=n - 1) * std / math.sqrt(n))
    return float(mean), ci, n


def get_rate_e7(e7: dict, pareto_key: str, nominal_bits: float) -> float:
    """Get realized rate from E7 JSON for a pareto condition key."""
    entry = e7.get("pareto", {}).get(pareto_key, {})
    bits = entry.get("rate", {}).get("bits_per_transition", float("nan"))
    return float(bits)


def plot_pareto_two_view(pareto: dict, e7: dict, outdir: Path) -> None:
    """Generate the two-panel Pareto figure."""
    _setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8), sharey=True)
    ax_nominal, ax_realized = axes

    # y-axis limits: determine from all data
    all_y_vals = []
    for budget in BUDGETS:
        for key in budget["ddcl_keys"]:
            mean, _, _ = get_perf(pareto, "ddcl", key)
            if not np.isnan(mean):
                all_y_vals.append(mean)
        for method_type, key in [("fsq", budget["fsq_key"]), ("vq", budget["vq_key"])]:
            mean, _, _ = get_perf(pareto, method_type, key)
            if not np.isnan(mean):
                all_y_vals.append(mean)
    y_max = max(all_y_vals) * 1.12 if all_y_vals else 1.0

    # ── Panel A: Nominal capacity ─────────────────────────────────────────────
    for budget in BUDGETS:
        nom_bits = budget["bits"]

        # DDCL λ-sweep curve (all λ values at this scale)
        ddcl_x, ddcl_y, ddcl_ci = [], [], []
        for key in budget["ddcl_keys"]:
            mean, ci, n = get_perf(pareto, "ddcl", key)
            if not np.isnan(mean):
                ddcl_x.append(nom_bits + np.random.default_rng(hash(key) % 2**32).uniform(-4, 4))
                ddcl_y.append(mean)
                ddcl_ci.append(ci)

        # Plot DDCL λ-sweep as connected curve with markers
        if ddcl_x:
            ax_nominal.scatter(
                ddcl_x, ddcl_y,
                color=PALETTE["ddcl"], marker=MARKERS["ddcl"],
                s=30, alpha=0.6, linewidths=0.4, zorder=3,
            )
            ax_nominal.errorbar(
                ddcl_x, ddcl_y, yerr=ddcl_ci,
                fmt="none", ecolor=PALETTE["ddcl"], elinewidth=0.7, capsize=2, zorder=3,
            )

        # FSQ
        fsq_mean, fsq_ci, _ = get_perf(pareto, "fsq", budget["fsq_key"])
        if not np.isnan(fsq_mean):
            ax_nominal.errorbar(
                nom_bits, fsq_mean, yerr=fsq_ci,
                fmt=MARKERS["fsq"], color=PALETTE["fsq"],
                markersize=8, markeredgewidth=0.5,
                elinewidth=0.8, capsize=2, zorder=4,
            )

        # VQ
        vq_mean, vq_ci, _ = get_perf(pareto, "vq", budget["vq_key"])
        if not np.isnan(vq_mean):
            ax_nominal.errorbar(
                nom_bits, vq_mean, yerr=vq_ci,
                fmt=MARKERS["vq"], color=PALETTE["vq"],
                markersize=8, markeredgewidth=0.5,
                elinewidth=0.8, capsize=2, zorder=4,
            )

    ax_nominal.set_xlabel("Nominal capacity (bits)")
    ax_nominal.set_ylabel(Y_LABEL)
    ax_nominal.set_xlim(200, 430)
    ax_nominal.set_ylim(0, y_max)
    ax_nominal.set_title("(A) Equal nominal budget", fontsize=FONT, pad=3)

    # Budget level markers
    for budget in BUDGETS:
        ax_nominal.axvline(budget["bits"], color="#AAAAAA", lw=0.7, ls=":", alpha=0.7, zorder=1)
        ax_nominal.text(budget["bits"] + 4, 0.02, f"{budget['bits']:.0f} b",
                        fontsize=FONT - 2, color="#777777", va="bottom")

    # ── Panel B: Realized rate on D_ref ───────────────────────────────────────
    for budget in BUDGETS:
        nom_bits = budget["bits"]
        e7_ddcl_keys = budget["e7_ddcl_keys"]

        # DDCL λ-sweep: realized rate (x) + on-policy perf (y)
        ddcl_x, ddcl_y, ddcl_ci = [], [], []
        for pareto_key, e7_key in e7_ddcl_keys.items():
            mean, ci, _ = get_perf(pareto, "ddcl", pareto_key)
            bits = get_rate_e7(e7, e7_key, nom_bits)
            if not np.isnan(mean) and not np.isnan(bits):
                ddcl_x.append(bits)
                ddcl_y.append(mean)
                ddcl_ci.append(ci)

        if ddcl_x:
            # Sort by realized rate for cleaner curve
            order = np.argsort(ddcl_x)
            ddcl_x = [ddcl_x[i] for i in order]
            ddcl_y = [ddcl_y[i] for i in order]
            ddcl_ci = [ddcl_ci[i] for i in order]
            ax_realized.plot(ddcl_x, ddcl_y, "-", color=PALETTE["ddcl"],
                             lw=0.8, alpha=0.5, zorder=2)
            ax_realized.scatter(ddcl_x, ddcl_y, color=PALETTE["ddcl"],
                                marker=MARKERS["ddcl"], s=30, alpha=0.7,
                                linewidths=0.4, zorder=3)
            ax_realized.errorbar(ddcl_x, ddcl_y, yerr=ddcl_ci,
                                 fmt="none", ecolor=PALETTE["ddcl"],
                                 elinewidth=0.7, capsize=2, zorder=3)

        # FSQ
        fsq_mean, fsq_ci, _ = get_perf(pareto, "fsq", budget["fsq_key"])
        fsq_bits = get_rate_e7(e7, budget["e7_fsq_key"], nom_bits)
        if not np.isnan(fsq_mean) and not np.isnan(fsq_bits):
            ax_realized.errorbar(
                fsq_bits, fsq_mean, yerr=fsq_ci,
                fmt=MARKERS["fsq"], color=PALETTE["fsq"],
                markersize=8, markeredgewidth=0.5,
                elinewidth=0.8, capsize=2, zorder=4,
            )

        # VQ
        vq_mean, vq_ci, _ = get_perf(pareto, "vq", budget["vq_key"])
        vq_bits = get_rate_e7(e7, budget["e7_vq_key"], nom_bits)
        if not np.isnan(vq_mean) and not np.isnan(vq_bits):
            ax_realized.errorbar(
                vq_bits, vq_mean, yerr=vq_ci,
                fmt=MARKERS["vq"], color=PALETTE["vq"],
                markersize=8, markeredgewidth=0.5,
                elinewidth=0.8, capsize=2, zorder=4,
            )

    ax_realized.set_xlabel("Realized code entropy on D_ref (bits/transition)")
    ax_realized.set_title("(B) Realized rate (shared probe set)", fontsize=FONT, pad=3)
    ax_realized.set_ylim(0, y_max)

    # Budget level markers (realized rate positions roughly)
    for budget in BUDGETS:
        ax_realized.axvline(budget["bits"], color="#AAAAAA", lw=0.7, ls=":", alpha=0.5, zorder=1)

    # Shared legend on Panel A
    legend_handles = [
        mlines.Line2D([], [], color=PALETTE["ddcl"], marker=MARKERS["ddcl"],
                      markersize=6, linewidth=0.8, label="DDCL (λ-sweep)"),
        mlines.Line2D([], [], color=PALETTE["fsq"], marker=MARKERS["fsq"],
                      markersize=6, linewidth=0, label="FSQ-CE"),
        mlines.Line2D([], [], color=PALETTE["vq"], marker=MARKERS["vq"],
                      markersize=6, linewidth=0, label="VQ-CE"),
    ]
    ax_nominal.legend(handles=legend_handles, loc="upper left",
                      fontsize=FONT - 1, borderpad=0.4)

    fig.tight_layout(pad=0.8)

    stem = "fig3_pareto"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outdir / stem}.png / .pdf")


def print_summary(pareto: dict, e7: dict) -> None:
    print("\n── Matched-capacity comparison ───────────────────────────────────────")
    for budget in BUDGETS:
        nom = budget["bits"]
        print(f"\n  Budget {nom:.0f} bits:")
        print(f"  {'Method':<28} {'Prec@0.02':>10}  {'Realized H':>12}  n")
        # DDCL best λ
        best_p, best_key = -1, None
        for key in budget["ddcl_keys"]:
            mean, _, _ = get_perf(pareto, "ddcl", key)
            if not np.isnan(mean) and mean > best_p:
                best_p, best_key = mean, key
        if best_key:
            mean, ci, n = get_perf(pareto, "ddcl", best_key)
            e7_key = budget["e7_ddcl_keys"].get(best_key, "")
            bits = get_rate_e7(e7, e7_key, nom)
            print(f"  DDCL (best λ={best_key.split('l')[1]})       {mean:.3f}±{ci:.3f}  {bits:>12.1f}  {n}")
        # FSQ
        mean, ci, n = get_perf(pareto, "fsq", budget["fsq_key"])
        bits = get_rate_e7(e7, budget["e7_fsq_key"], nom)
        print(f"  FSQ-{budget['fsq_key']:<22}  {mean:.3f}±{ci:.3f}  {bits:>12.1f}  {n}")
        # VQ
        mean, ci, n = get_perf(pareto, "vq", budget["vq_key"])
        bits = get_rate_e7(e7, budget["e7_vq_key"], nom)
        print(f"  VQ-{budget['vq_key']:<23}  {mean:.3f}±{ci:.3f}  {bits:>12.1f}  {n}")


def main():
    date_str = datetime.now().strftime("%Y%m%d")
    outdir   = OUT_BASE / f"paper_figs_{date_str}"

    print("Loading Pareto data ...")
    pareto = load_pareto()
    e7     = load_e7()
    print_summary(pareto, e7)

    print("\nGenerating Fig 3 ...")
    plot_pareto_two_view(pareto, e7, outdir)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
