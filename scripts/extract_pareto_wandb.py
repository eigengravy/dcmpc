#!/usr/bin/env python3
"""Extract quantizer Pareto sweep results from W&B and save canonical JSON + plots.

Fetches all runs from ddcl_mbrl_toy_pareto, groups by (method_family, config_key),
aggregates over seeds, and produces:
  - private/Metrics/Toy/QuantizerPareto/pareto_sweep_aggregated.json
  - private/Plots/quantizer_pareto_<date>/  (PDF + PNG figures)

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/extract_pareto_wandb.py
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

tmp_dir = Path(os.getenv("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_dir / "matplotlib"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import wandb

# ── Config ────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "ddcl_mbrl_toy_pareto"
OUT_METRICS   = Path("private/Metrics/Toy/QuantizerPareto")
OUT_PLOTS     = Path(f"private/Plots/quantizer_pareto_{datetime.now().strftime('%Y%m%d')}")
PAPER_FIGS    = Path("../DDCL_MBRL_WM_RLC_Workshop/Figures")

# Paper style
COL_W, FONT_SIZE = 3.25, 8
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE-1, "ytick.labelsize": FONT_SIZE-1,
    "legend.fontsize": FONT_SIZE-1, "lines.linewidth": 1.2, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "legend.frameon": False,
})

# Paul Tol bright palette per quantizer family
FAMILY_COLOR = {
    "ddcl":  "#332288",   # indigo
    "fsq":   "#4477AA",   # blue
    "vq":    "#66CCEE",   # cyan
}
FAMILY_MARKER = {"ddcl": "*", "fsq": "^", "vq": "v"}
FAMILY_LABEL  = {"ddcl": "DDCL cosine", "fsq": "FSQ CE", "vq": "VQ CE"}

# Canonical paper points (from 5-seed heldout eval) drawn as large markers for reference
CANONICAL = {
    "DDCL cosine": (212.1, 0.744, "#332288", "*", 120),
    "DDCL MSE":    (199.0, 0.584, "#228833", "X", 80),
    "DDCL CE":     (183.1, 0.308, "#EE6677", "o", 60),
    "VQ CE":       (138.2, 0.228, "#66CCEE", "v", 80),
    "FSQ CE":      (127.6, 0.200, "#4477AA", "^", 60),
    "Cont. MSE":   (None,  0.592, "#BBBBBB", "s", 60),  # no rate
}

# ── Fetch runs ────────────────────────────────────────────────────────────────
def get_family(config: dict) -> str:
    q = config.get("agent", {}).get("quantizer", "")
    if q == "ddcl": return "ddcl"
    if q == "fsq":  return "fsq"
    if q == "vq":   return "vq"
    return "other"

def config_key(config: dict, family: str) -> str:
    a = config.get("agent", {})
    if family == "ddcl":
        return f"s{a.get('ddcl_scale','?')}_d{a.get('ddcl_delta','?')}_l{a.get('ddcl_lambda','?')}"
    if family == "fsq":
        return f"fsq_{a.get('fsq_levels','?')}"
    if family == "vq":
        return f"vq_{a.get('vq_codebook_size','?')}"
    return "other"

print(f"Fetching runs from W&B project: {WANDB_PROJECT}")
api = wandb.Api(timeout=60)
runs = list(api.runs(WANDB_PROJECT, per_page=200))
print(f"Found {len(runs)} runs")

# Group: {family: {config_key: {seed: metrics_dict}}}
groups: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

for r in runs:
    if r.state != "finished":
        continue
    family = get_family(r.config)
    if family == "other":
        continue
    key = config_key(r.config, family)
    seed = r.config.get("seed", -1)
    eb = r.summary.get("eval_best/", {})

    metrics = {
        "episodic_success":                  eb.get("episodic_success"),
        "episodic_precision_success_0p02":   eb.get("episodic_precision_success_0p02"),
        "episodic_precision_success_0p04":   eb.get("episodic_precision_success_0p04"),
        "episodic_return":                   eb.get("episodic_return"),
        "rate/empirical_entropy_bits_per_transition": eb.get("rate/empirical_entropy_bits_per_transition"),
        # Use ddcl_loss_bits_per_dim × n_total_dims for DDCL (unsigned formula,
        # matches DDCL paper and training comm_loss). For VQ/FSQ this is None and
        # the aggregate function falls back to max_bits_per_transition via None-skip.
        # Runs before 2026-05-21 logged rate/allocated_bits_per_transition with
        # the inflated signed formula; use ddcl_loss_bits_per_dim instead.
        "rate/ddcl_loss_bits_per_dim":                eb.get("rate/ddcl_loss_bits_per_dim"),
        "rate/ddcl_signed_bits_per_transition":       eb.get("rate/ddcl_signed_bits_per_transition"),
        "rate/allocated_bits_per_transition":         eb.get("rate/allocated_bits_per_transition"),
        "codebook/usage_percent":            eb.get("active_percent"),
        # Store config for label
        "_ddcl_scale":  r.config.get("agent", {}).get("ddcl_scale"),
        "_ddcl_lambda": r.config.get("agent", {}).get("ddcl_lambda"),
        "_fsq_levels":  r.config.get("agent", {}).get("fsq_levels"),
        "_vq_size":     r.config.get("agent", {}).get("vq_codebook_size"),
    }
    groups[family][key][seed] = metrics

# ── Aggregate over seeds ──────────────────────────────────────────────────────
def agg_seeds(seed_dict: dict) -> dict:
    vals: dict[str, list] = defaultdict(list)
    for m in seed_dict.values():
        for k, v in m.items():
            if v is not None and not k.startswith("_"):
                vals[k].append(v)
    out = {}
    for k, vs in vals.items():
        arr = np.array(vs, dtype=float)
        out[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": len(arr)}
    # Store config from first seed
    first = next(iter(seed_dict.values()))
    for k, v in first.items():
        if k.startswith("_"):
            out[k] = v
    out["_n_seeds"] = len(seed_dict)
    return out

aggregated: dict[str, dict[str, dict]] = {}
for family, configs in groups.items():
    aggregated[family] = {}
    for key, seeds in configs.items():
        agg = agg_seeds(seeds)

        # ── Correct DDCL allocated_bits to use unsigned formula ──────────────
        # Runs before 2026-05-21 logged rate/allocated_bits_per_transition with
        # the signed formula log2(2|z|/δ+1). Compute the correct value from
        # the unsigned per-dim metric: loss_per_dim × n_total_dims.
        if family == "ddcl":
            loss_pd = agg.get("rate/ddcl_loss_bits_per_dim")
            signed_total_agg = agg.get("rate/ddcl_signed_bits_per_transition")
            signed_pd_agg = agg.get("rate/ddcl_signed_bits_per_dim")
            if (
                loss_pd is not None
                and signed_total_agg is not None
                and signed_pd_agg is not None
                and isinstance(loss_pd, dict)
                and isinstance(signed_pd_agg, dict)
                and isinstance(signed_total_agg, dict)
                and signed_pd_agg["mean"] > 1e-9
            ):
                n_total_dims = round(signed_total_agg["mean"] / signed_pd_agg["mean"])
                corrected_mean = loss_pd["mean"] * n_total_dims
                corrected_std  = loss_pd["std"]  * n_total_dims
                agg["rate/allocated_bits_per_transition"] = {
                    "mean": corrected_mean, "std": corrected_std, "n": loss_pd["n"]
                }

        aggregated[family][key] = agg

OUT_METRICS.mkdir(parents=True, exist_ok=True)
json_path = OUT_METRICS / "pareto_sweep_aggregated.json"
with open(json_path, "w") as f:
    json.dump(aggregated, f, indent=2, default=str)
print(f"Saved aggregated metrics → {json_path}")

# ── Plot: Pareto frontier ─────────────────────────────────────────────────────
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

def make_pareto_fig(x_metric: str, y_metric: str, ylabel: str, stem: str):
    fig, ax = plt.subplots(figsize=(COL_W * 1.15, 2.6))

    for family in ["fsq", "vq", "ddcl"]:
        if family not in aggregated:
            continue
        xs, ys, labels = [], [], []
        for key, agg in aggregated[family].items():
            xm = agg.get(x_metric, {})
            ym = agg.get(y_metric, {})
            if not xm or not ym:
                continue
            xv, yv = xm.get("mean"), ym.get("mean")
            if xv is None or yv is None:
                continue
            xs.append(xv); ys.append(yv)
            # Build short label
            if family == "ddcl":
                lbl = f"λ={agg.get('_ddcl_lambda','?')}"
            elif family == "fsq":
                lvls = agg.get('_fsq_levels', '?')
                lbl = f"{lvls}"
            else:
                lbl = f"C={agg.get('_vq_size','?')}"
            labels.append(lbl)

        if not xs:
            continue
        color  = FAMILY_COLOR[family]
        marker = FAMILY_MARKER[family]
        ax.scatter(xs, ys, c=color, marker=marker, s=25, alpha=0.75,
                   label=FAMILY_LABEL[family], zorder=3)

    # Canonical paper points as larger reference markers
    for name, (rate, prec, color, marker, sz) in CANONICAL.items():
        if rate is None:
            continue
        ax.scatter([rate], [prec], c=color, marker=marker, s=sz,
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(name, (rate, prec), textcoords="offset points",
                    xytext=(4, 2), fontsize=FONT_SIZE - 2, color=color)

    ax.set_xlabel("Empirical entropy (bits/transition)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.08)
    ax.legend(loc="upper left", markerscale=1.4)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT_PLOTS / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  → {path}")
    plt.close(fig)
    return OUT_PLOTS / f"{stem}.pdf"

pdf_prec = make_pareto_fig(
    "rate/empirical_entropy_bits_per_transition",
    "episodic_precision_success_0p02",
    "Precision success @ 0.02",
    "pareto_precision_0p02_vs_entropy_sweep",
)
make_pareto_fig(
    "rate/empirical_entropy_bits_per_transition",
    "episodic_success",
    "Success",
    "pareto_success_vs_entropy_sweep",
)

# Copy the precision Pareto to paper Figures dir as Fig 3
import shutil
paper_fig3 = PAPER_FIGS / "03_toy_rate_precision_tradeoff.pdf"
shutil.copy(pdf_prec, paper_fig3)
print(f"\nUpdated paper Figure 3 → {paper_fig3}")
print("Done.")
