#!/usr/bin/env python3
"""Unified paper-figure generator for DDCL-MBRL RLC 2026 workshop paper.

Toy figures (Fig 03–05) are generated from the canonical corrected JSON files.
Transfer figures (Fig 06–08) are fetched live from Weights & Biases.

Outputs (PNG + PDF) are written to DDCL_MBRL_WM_RLC_Workshop/Figures/ by default.

Usage:
    # All figures (toy from JSON, transfer from W&B):
    conda run -n ddcl_mbrl python scripts/make_paper_figures.py

    # Toy only (no W&B needed):
    conda run -n ddcl_mbrl python scripts/make_paper_figures.py --toy-only

    # Transfer only:
    conda run -n ddcl_mbrl python scripts/make_paper_figures.py --transfer-only

    # Custom output directory:
    conda run -n ddcl_mbrl python scripts/make_paper_figures.py --outdir /path/to/Figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# ── Matplotlib/cache setup before any import ─────────────────────────────────
_tmp = Path(os.getenv("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_tmp / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp / "xdg-cache"))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

# ── Repo paths ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "results" / "plotting"))
from stats_utils import compute_stats  # noqa: E402

METRICS_DIR = REPO / "private" / "Metrics" / "Toy" / "Corrected Reevaluation"
DEFAULT_OUTDIR = REPO.parent / "DDCL_MBRL_WM_RLC_Workshop" / "Figures"

# ── W&B credentials (read from environment; never hard-code secrets) ──────────
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "f20210966")
WANDB_PROJECT = "dcmpc"  # transfer-learning runs

# ── Paper layout (RLC 2026, 10 pt, two-column) ────────────────────────────────
COL_W    = 3.25   # single-column width, inches
FULL_W   = 6.75   # full-page width, inches
FONT_PT  = 8      # body text size

# ── Paul Tol "bright" colorblind-safe palette ─────────────────────────────────
COLORS: dict[str, str] = {
    "continuous_mse": "#BBBBBB",   # grey
    "fsq_ce":         "#4477AA",   # blue
    "vq_ce":          "#66CCEE",   # cyan
    "ddcl_ce":        "#EE6677",   # red
    "ddcl_soft_ce":   "#CCBB44",   # yellow
    "ddcl_mse":       "#228833",   # green
    "ddcl_cosine":    "#EE7733",   # orange
}
MARKERS: dict[str, str] = {
    "continuous_mse": "s",
    "fsq_ce":         "^",
    "vq_ce":          "v",
    "ddcl_ce":        "o",
    "ddcl_soft_ce":   "D",
    "ddcl_mse":       "X",
    "ddcl_cosine":    "*",
}
LABELS: dict[str, str] = {
    "continuous_mse": "Cont-MSE",
    "fsq_ce":         "FSQ-CE",
    "vq_ce":          "VQ-CE",
    "ddcl_ce":        "DDCL-CE",
    "ddcl_soft_ce":   "DDCL-SCE",
    "ddcl_mse":       "DDCL-MSE",
    "ddcl_cosine":    "DDCL-Cos",
}
# Canonical method display order
METHOD_ORDER = [
    "continuous_mse", "fsq_ce", "vq_ce",
    "ddcl_ce", "ddcl_soft_ce", "ddcl_mse", "ddcl_cosine",
]

# Transfer-only method order / labels
TRANSFER_METHODS = ["FSQ-CE", "VQ-CE", "DDCL-Cos"]
TRANSFER_COLORS  = {"FSQ-CE": "#4477AA", "VQ-CE": "#66CCEE", "DDCL-Cos": "#EE7733"}
TRANSFER_MARKERS = {"FSQ-CE": "^",       "VQ-CE": "v",        "DDCL-Cos": "*"}

# Transfer environment keys and display labels
ENV_KEYS   = ["walker/walk", "reacher/hard", "dog/run", "humanoid/walk", "mw-button-press"]
ENV_LABELS = ["Walker\nWalk", "Reacher\nHard", "Dog\nRun", "Humanoid\nWalk", "Button\nPress"]
ENV_SHORT  = {
    "walker/walk":     "Walker",
    "reacher/hard":    "Reacher",
    "dog/run":         "Dog",
    "humanoid/walk":   "Humanoid",
    "mw-button-press": "Button",
}
ENV_MARKERS = {
    "walker/walk":     "o",
    "reacher/hard":    "s",
    "dog/run":         "D",
    "humanoid/walk":   "^",
    "mw-button-press": "P",
}
BEST_LAMBDA = {
    "walker/walk":     0.001,
    "reacher/hard":    0.01,
    "dog/run":         0.001,
    "humanoid/walk":   0.01,
    "mw-button-press": 0.01,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Style
# ═══════════════════════════════════════════════════════════════════════════════

def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "pdf.fonttype":        42,
        "ps.fonttype":         42,
        "font.size":           FONT_PT,
        "axes.titlesize":      FONT_PT,
        "axes.labelsize":      FONT_PT,
        "xtick.labelsize":     FONT_PT - 1,
        "ytick.labelsize":     FONT_PT - 1,
        "legend.fontsize":     FONT_PT - 1,
        "lines.linewidth":     1.2,
        "axes.linewidth":      0.7,
        "xtick.major.width":   0.7,
        "ytick.major.width":   0.7,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.alpha":          0.3,
        "grid.linewidth":      0.5,
        "legend.frameon":      False,
        "legend.handlelength": 1.2,
    })


def savefig(outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig_path = outdir / f"{stem}.{ext}"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"  → {outdir / stem}.png / .pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CI utilities — the key fix for blown-out error bars
# ═══════════════════════════════════════════════════════════════════════════════

def clip_ci(mean: float, ci_half: float, lo: float = 0.0, hi: float = 1.0):
    """Return asymmetric (lower_err, upper_err) clipped to valid range [lo, hi].

    Matplotlib expects yerr as [[lower_errors], [upper_errors]] where each
    value is the distance from the mean to the error bar tip (always >= 0).

    Without clipping a CI of 0.68 on a mean of 0.41 would extend to -0.27,
    which is meaningless for a probability/success metric.
    """
    if np.isnan(ci_half):
        return np.nan, np.nan
    lower_err = min(ci_half, mean - lo)
    upper_err = min(ci_half, hi - mean)
    return max(0.0, lower_err), max(0.0, upper_err)


def stats_from_values(values: list) -> dict:
    return compute_stats(values)


def ci95_half(values: list) -> float:
    st = compute_stats(values)
    return st["ci95"]


# ═══════════════════════════════════════════════════════════════════════════════
# Toy data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_toy_data() -> dict:
    """Load all canonical toy JSON files and merge into one dict keyed by method."""
    data: dict[str, dict] = {}

    # Main 5-seed heldout eval (all methods except SCE)
    main_json = METRICS_DIR / "wandb_toy_all_methods_5seed_heldout_20260517.json"
    if main_json.exists():
        with open(main_json) as f:
            data.update(json.load(f))
        print(f"  Loaded main toy data from {main_json.name}")
    else:
        warnings.warn(f"Main toy JSON not found: {main_json}")

    # DDCL-SCE heldout eval — merge/override
    sce_jsons = sorted(METRICS_DIR.glob("wandb_toy_soft_ce_*heldout*.json"))
    if sce_jsons:
        with open(sce_jsons[-1]) as f:
            sce = json.load(f)
        data.update(sce)
        print(f"  Loaded DDCL-SCE from {sce_jsons[-1].name}")
    else:
        warnings.warn("DDCL-SCE heldout JSON not found; SCE will be absent from toy plots.")

    return data


def load_stoch_ablation_data() -> dict:
    """Load stochasticity ablation JSON + main heldout for (d,d) conditions."""
    data: dict[str, dict] = {}

    main_json = METRICS_DIR / "wandb_toy_all_methods_5seed_heldout_20260517.json"
    if main_json.exists():
        with open(main_json) as f:
            raw = json.load(f)
        # Rename canonical (d,d) keys to match ablation convention
        if "ddcl_ce" in raw:
            data["ddcl_ce_dd"] = raw["ddcl_ce"]
        if "ddcl_cosine" in raw:
            data["ddcl_cos_dd"] = raw["ddcl_cosine"]

    stoch_jsons = sorted(METRICS_DIR.glob("wandb_toy_stoch_ablation_*heldout*.json"))
    if stoch_jsons:
        with open(stoch_jsons[-1]) as f:
            stoch = json.load(f)
        data.update(stoch)
        print(f"  Loaded stoch ablation data from {stoch_jsons[-1].name}")
    else:
        warnings.warn("Stoch ablation JSON not found; ablation bars will be empty.")

    return data


def _get_vals(data: dict, method: str, metric: str) -> list[float]:
    try:
        return data[method]["metrics"][metric]["values"]
    except KeyError:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Transfer data loading (W&B)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_transfer_runs() -> list[dict]:
    """Fetch finished transfer runs from W&B via the Python SDK."""
    try:
        import wandb
    except ImportError:
        warnings.warn("wandb not installed; transfer figures will be skipped.")
        return []

    os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY)
    try:
        api = wandb.Api(api_key=WANDB_API_KEY, timeout=60)
        runs = list(api.runs(
            path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"state": "finished"},
            per_page=200,
        ))
        print(f"  Fetched {len(runs)} finished runs from W&B {WANDB_ENTITY}/{WANDB_PROJECT}")
    except Exception as exc:
        warnings.warn(f"W&B fetch failed: {exc}")
        return []

    records = []
    for run in runs:
        cfg = dict(run.config or {})
        sm  = dict(run.summary or {})

        env_name  = _cfg_val(cfg, "env_name")
        task_name = _cfg_val(cfg, "task_name")
        seed      = _cfg_val(cfg, "seed")
        env_key   = f"{env_name}/{task_name}".rstrip("/") if env_name else ""
        if env_key not in ENV_KEYS:
            continue

        agent = _cfg_val(cfg, "agent") or {}
        if not isinstance(agent, dict):
            agent = {}
        quantizer = str(agent.get("quantizer", "")).lower()
        lam       = agent.get("ddcl_lambda")

        if quantizer == "fsq":
            method = "FSQ-CE"
        elif quantizer == "vq":
            method = "VQ-CE"
        elif quantizer == "ddcl":
            method = "DDCL-Cos"
        else:
            continue

        # Prefer eval_best/, fall back to eval/
        eb = sm.get("eval_best/", {}) or {}
        ev = sm.get("eval/", {})      or {}
        src_d = eb if (eb.get("normalized_return") or eb.get("episodic_return")
                       or eb.get("episodic_success")) else ev

        nr        = src_d.get("normalized_return")
        raw_ret   = src_d.get("episodic_return")
        suc       = src_d.get("episodic_success")
        alloc_b   = src_d.get("rate/allocated_bits_per_transition")
        emp_b     = src_d.get("rate/empirical_entropy_bits_per_transition")
        usage     = src_d.get("codebook/usage_percent")
        nom_bits  = src_d.get("rate/max_bits_per_transition")

        # Normalise raw return if normalized_return absent
        if nr is None and raw_ret is not None:
            nr = raw_ret / 1000.0

        # Correct inflated allocated_bits for DDCL runs before 2026-05-21:
        # use ddcl_loss_bits_per_dim × n_total_dims instead.
        if method == "DDCL-Cos" and alloc_b is not None:
            loss_per_dim = src_d.get("rate/ddcl_loss_bits_per_dim")
            signed_total = src_d.get("rate/ddcl_signed_bits_per_transition")
            signed_per   = src_d.get("rate/ddcl_signed_bits_per_dim")
            if (loss_per_dim is not None and signed_total is not None
                    and signed_per is not None and signed_per > 0):
                n_dims   = round(signed_total / signed_per)
                alloc_b  = loss_per_dim * n_dims

        records.append(dict(
            env=env_key, method=method, lam=lam, seed=seed,
            nr=nr, suc=suc,
            alloc_bits=alloc_b, emp_bits=emp_b,
            usage_pct=usage, nom_bits=nom_bits,
        ))

    print(f"  Parsed {len(records)} transfer records across {set(r['env'] for r in records)}")
    return records


def _cfg_val(cfg: dict, key: str):
    v = cfg.get(key)
    if isinstance(v, dict):
        return v.get("value", v)
    return v


def group_transfer_records(records: list[dict]) -> dict:
    """Group records into {(env, method): {metric_key: [values]}}."""
    from collections import defaultdict
    g: dict = defaultdict(lambda: {
        "nr": [], "suc": [], "alloc_bits": [], "emp_bits": [], "usage_pct": []
    })
    for r in records:
        k = (r["env"], r["method"])
        for field in ("nr", "suc", "alloc_bits", "emp_bits", "usage_pct"):
            v = r.get(field)
            if v is not None and not np.isnan(float(v)):
                g[k][field].append(float(v))
    return dict(g)


def select_best_ddcl_lambda(records: list[dict]) -> list[dict]:
    """Keep only best-lambda seeds for DDCL; keep all FSQ/VQ seeds."""
    out = []
    for r in records:
        if r["method"] != "DDCL-Cos":
            out.append(r)
        elif r.get("lam") == BEST_LAMBDA.get(r["env"]):
            out.append(r)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 03: Toy Rate-Precision Pareto (two panels)
# ═══════════════════════════════════════════════════════════════════════════════

def fig03_toy_pareto(toy_data: dict, outdir: Path) -> None:
    """Two-panel scatter: empirical entropy bits vs precision@0.02.
    Panel A: all methods; Panel B: baselines vs DDCL-Cos.
    """
    X_METRIC = "rate/empirical_entropy_bits_per_transition"
    Y_METRIC = "episodic_precision_success_0p02"

    methods_a = ["vq_ce", "fsq_ce", "ddcl_ce", "ddcl_soft_ce", "ddcl_mse", "ddcl_cosine"]
    methods_b = ["vq_ce", "fsq_ce", "ddcl_cosine"]

    # Gather per-method stats
    points: dict[str, dict] = {}
    for m in methods_a:
        xv = _get_vals(toy_data, m, X_METRIC)
        yv = _get_vals(toy_data, m, Y_METRIC)
        if not xv or not yv:
            continue
        sx = compute_stats(xv)
        sy = compute_stats(yv)
        lo_y, hi_y = clip_ci(sy["mean"], sy["ci95"])
        lo_x, hi_x = clip_ci(sx["mean"], sx["ci95"], lo=0, hi=np.inf)
        points[m] = dict(
            x=sx["mean"], y=sy["mean"],
            xerr=[[lo_x], [hi_x]],
            yerr=[[lo_y], [hi_y]],
            x_vals=sx["values"], y_vals=sy["values"],
        )

    # Cont-MSE: no rate axis — show as horizontal reference line
    cont_vals = _get_vals(toy_data, "continuous_mse", Y_METRIC)
    cont_y = float(np.mean(cont_vals)) if cont_vals else None

    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2.2, 2.5))

    for ax_i, (ax, panel_methods, title) in enumerate(zip(
        axes,
        [methods_a, methods_b],
        ["(a) All methods", "(b) Baselines vs. DDCL-Cos"],
    )):
        if cont_y is not None:
            ax.axhline(
                cont_y,
                color=COLORS["continuous_mse"], lw=0.9, ls="--", alpha=0.75,
                label=LABELS["continuous_mse"], zorder=1,
            )

        for m in panel_methods:
            if m not in points:
                continue
            p    = points[m]
            col  = COLORS[m]
            mkr  = MARKERS[m]
            ms   = 8 if (m == "ddcl_cosine" and ax_i == 1) else 5

            # Individual seed dots (translucent)
            for xv, yv in zip(p["x_vals"], p["y_vals"]):
                ax.scatter(xv, yv, c=col, marker=mkr, s=12, alpha=0.30,
                           linewidths=0, zorder=2)

            # Mean ± clipped 95% CI
            ax.errorbar(
                p["x"], p["y"],
                xerr=p["xerr"], yerr=p["yerr"],
                fmt=mkr, color=col, markersize=ms,
                markeredgewidth=0.5 if ms > 5 else 0,
                capsize=2.5, capthick=0.7, elinewidth=0.8,
                label=LABELS[m], zorder=4,
            )

        # Annotation: DDCL-Cos vs VQ-CE ratio (panel B only)
        if ax_i == 1 and "ddcl_cosine" in points and "vq_ce" in points:
            cos_y = points["ddcl_cosine"]["y"]
            vq_y  = points["vq_ce"]["y"]
            cos_x = points["ddcl_cosine"]["x"]
            if vq_y > 0:
                ratio = cos_y / vq_y
                ax.annotate(
                    f"{ratio:.1f}× vs VQ-CE",
                    xy=(cos_x, cos_y),
                    xytext=(cos_x - 22, cos_y + 0.12),
                    fontsize=FONT_PT - 2, color=COLORS["ddcl_cosine"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["ddcl_cosine"], lw=0.7),
                    ha="center",
                )

        ax.set_xlabel("Empirical entropy (bits / transition)")
        ax.set_ylabel("Precision success @ 0.02" if ax_i == 0 else "")
        ax.set_ylim(-0.02, 1.12)
        ax.set_title(title, fontsize=FONT_PT)
        ax.legend(loc="upper left", fontsize=FONT_PT - 2, borderpad=0.3)

    fig.tight_layout()
    savefig(outdir, "03_toy_rate_precision_tradeoff")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 04: DDCL objective ablation bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig04_objective_ablation(toy_data: dict, outdir: Path) -> None:
    """Two-panel bar chart: DDCL-only methods, Success and Precision@0.02."""
    methods = ["ddcl_ce", "ddcl_soft_ce", "ddcl_mse", "ddcl_cosine"]
    metrics_cfg = [
        ("episodic_success",                "Success"),
        ("episodic_precision_success_0p02", "Precision success @ 0.02"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.72, 2.3), sharey=False)
    bar_w = 0.55
    x = np.arange(len(methods))

    for ax, (metric_key, ylabel) in zip(axes, metrics_cfg):
        for i, m in enumerate(methods):
            vals = _get_vals(toy_data, m, metric_key)
            if not vals:
                ax.bar(i, 0, bar_w, color=COLORS[m], alpha=0.25, zorder=2)
                continue

            st   = compute_stats(vals)
            mean = st["mean"]
            lo_e, hi_e = clip_ci(mean, st["ci95"])
            arr  = np.array(st["values"])

            ax.bar(i, mean, bar_w, color=COLORS[m], alpha=0.85, zorder=2)
            if not (np.isnan(lo_e) and np.isnan(hi_e)):
                ax.errorbar(
                    i, mean, yerr=[[lo_e], [hi_e]],
                    fmt="none", color="#222222",
                    capsize=3, capthick=0.8, elinewidth=0.9, zorder=3,
                )
            # Individual seed dots with deterministic jitter
            jitter = np.linspace(-0.14, 0.14, len(arr))
            ax.scatter(
                np.full_like(arr, i) + jitter, arr,
                s=9, color=COLORS[m], alpha=0.65, linewidths=0, zorder=4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods],
                           fontsize=FONT_PT - 1, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.18)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    # Annotation: cosine is 3.3× VQ-CE on prec@0.02
    ax_prec = axes[1]
    cos_vals = _get_vals(toy_data, "ddcl_cosine", "episodic_precision_success_0p02")
    if cos_vals:
        cos_mean = float(np.mean(cos_vals))
        cos_idx  = methods.index("ddcl_cosine")
        ax_prec.annotate(
            "3.3× vs VQ-CE",
            xy=(cos_idx, cos_mean),
            xytext=(cos_idx - 0.7, min(cos_mean + 0.22, 1.10)),
            fontsize=FONT_PT - 2, color=COLORS["ddcl_cosine"],
            arrowprops=dict(arrowstyle="-", color=COLORS["ddcl_cosine"], lw=0.8),
            ha="center",
        )

    fig.suptitle("DDCL dynamics objective ablation", fontsize=FONT_PT, y=1.01)
    fig.tight_layout()
    savefig(outdir, "04_ddcl_objective_ablation")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 05: Stochasticity ablation
# ═══════════════════════════════════════════════════════════════════════════════

def fig05_stoch_ablation(stoch_data: dict, outdir: Path) -> None:
    """2×2 panel: CE and Cosine objectives × Success and Prec@0.02.
    Each panel shows the stochastic/deterministic eval×targets conditions.
    """
    # (display_label, internal_key, color)
    CE_CONDS = [
        ("CE(s,s)", "ddcl_ce_ss", "#FFAAAA"),
        ("CE(s,d)", "ddcl_ce_sd", "#EE6677"),
        ("CE(d,s)", "ddcl_ce_ds", "#AA3377"),
        ("CE(d,d)", "ddcl_ce_dd", "#661133"),
    ]
    COS_CONDS = [
        ("Cos(s,d)", "ddcl_cos_sd",  "#88BBDD"),
        ("Cos(d,d)", "ddcl_cos_dd",  "#332288"),
    ]

    metrics = [
        ("episodic_success",                "Success"),
        ("episodic_precision_success_0p02", "Prec @ 0.02"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(COL_W * 2.0, 3.8), sharey=False)
    panel_cfgs = [
        (axes[0, 0], CE_CONDS,  metrics[0][0], metrics[0][1], "CE objective"),
        (axes[0, 1], CE_CONDS,  metrics[1][0], metrics[1][1], ""),
        (axes[1, 0], COS_CONDS, metrics[0][0], metrics[0][1], "Cosine objective"),
        (axes[1, 1], COS_CONDS, metrics[1][0], metrics[1][1], ""),
    ]
    bar_w = 0.55

    for ax, conditions, metric_key, ylabel, row_label in panel_cfgs:
        x = np.arange(len(conditions))
        for i, (disp_lbl, key, color) in enumerate(conditions):
            vals = _get_vals(stoch_data, key, metric_key)
            if not vals:
                ax.bar(i, 0, bar_w, color=color, alpha=0.25, zorder=2)
                ax.text(i, 0.04, "pending", ha="center", va="bottom",
                        fontsize=FONT_PT - 3, color="#999999", rotation=90)
                continue

            st   = compute_stats(vals)
            mean = st["mean"]
            lo_e, hi_e = clip_ci(mean, st["ci95"])
            arr  = np.array(st["values"])

            ax.bar(i, mean, bar_w, color=color, alpha=0.85, zorder=2)
            if not (np.isnan(lo_e) and np.isnan(hi_e)):
                ax.errorbar(
                    i, mean, yerr=[[lo_e], [hi_e]],
                    fmt="none", color="#222222",
                    capsize=3, capthick=0.8, elinewidth=0.9, zorder=3,
                )
            jitter = np.linspace(-0.10, 0.10, len(arr))
            ax.scatter(
                np.full_like(arr, i) + jitter, arr,
                s=7, color=color, alpha=0.60, linewidths=0, zorder=4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in conditions], fontsize=FONT_PT - 2)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
        if row_label:
            ax.set_title(row_label, fontsize=FONT_PT, loc="left", pad=2)

    fig.suptitle("Stochasticity ablation: det vs stoch eval / targets",
                 fontsize=FONT_PT, y=1.01)
    fig.tight_layout()
    savefig(outdir, "05_ddcl_stoch_ablation")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 06: Transfer performance — 5-panel grouped bar
# ═══════════════════════════════════════════════════════════════════════════════

def fig06_transfer_performance(groups: dict, outdir: Path) -> None:
    """One subplot per env, grouped bars for FSQ / VQ / DDCL-Cos.
    Primary metric: normalised return (DMControl) or success (MetaWorld).
    """
    fig, axes = plt.subplots(1, 5, figsize=(FULL_W, 2.2), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    bar_w = 0.6
    x_pos = np.arange(len(TRANSFER_METHODS))

    for ax, env_key, env_lbl in zip(axes, ENV_KEYS, ENV_LABELS):
        is_mw = "mw" in env_key

        for xi, method in enumerate(TRANSFER_METHODS):
            data = groups.get((env_key, method), {})
            vals = data.get("suc" if is_mw else "nr", [])
            if not vals:
                continue

            st   = compute_stats(vals)
            mean = st["mean"]
            lo_e, hi_e = clip_ci(mean, st["ci95"])
            arr  = np.array(st["values"])
            col  = TRANSFER_COLORS[method]

            ax.bar(xi, mean, width=bar_w, color=col, zorder=2,
                   label=method if ax is axes[0] else None)
            if not (np.isnan(lo_e) and np.isnan(hi_e)):
                ax.errorbar(
                    xi, mean, yerr=[[lo_e], [hi_e]],
                    fmt="none", ecolor="#333333",
                    capsize=3, capthick=0.9, elinewidth=0.9, zorder=3,
                )
            # Seed dots
            jitter = np.linspace(-0.12, 0.12, len(arr))
            ax.scatter(
                np.full_like(arr, xi) + jitter, arr,
                s=7, color=col, alpha=0.55, linewidths=0, zorder=4,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["FSQ", "VQ", "DDCL"], fontsize=FONT_PT - 1)
        ax.set_title(env_lbl, fontsize=FONT_PT, pad=3)
        ax.set_ylim(0, 1.12)
        ax.set_xlim(-0.6, 2.6)
        if ax is axes[0]:
            ax.set_ylabel("Normalised return / success")
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=TRANSFER_COLORS[m], label=m) for m in TRANSFER_METHODS]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.10), fontsize=FONT_PT - 1, frameon=False)

    fig.tight_layout(pad=0.6)
    savefig(outdir, "06_transfer_performance")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 07: Transfer rate efficiency scatter
# ═══════════════════════════════════════════════════════════════════════════════

def _pareto_mask(xs, ys) -> np.ndarray:
    """True where point (x, y) is on the upper-left Pareto frontier (lower x, higher y)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n  = len(xs)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if np.isnan(xs[i]) or np.isnan(ys[i]):
            mask[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            if np.isnan(xs[j]) or np.isnan(ys[j]):
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                mask[i] = False
                break
    return mask


def fig07_transfer_rate_efficiency(groups: dict, records: list[dict], outdir: Path) -> None:
    """Scatter: mean performance vs mean allocated bits.
    Each point is one (env, method); env encoded by marker shape.
    Pareto-optimal points per env are highlighted with black ring.
    """
    # Nominal bits per env (from FSQ/VQ runs, which don't apply a rate penalty)
    nom_bits_map: dict[str, float] = {}
    for r in records:
        nb = r.get("nom_bits")
        if nb is not None and not np.isnan(float(nb)):
            existing = nom_bits_map.get(r["env"])
            if existing is None or r["method"] in ("FSQ-CE", "VQ-CE"):
                nom_bits_map[r["env"]] = float(nb)

    # Small multiplicative x-dodge so FSQ-CE and VQ-CE (both at the nominal
    # budget) don't overlap. DDCL-Cos has its own actual x, so no dodge needed.
    # On a log scale a ×factor offset looks like a fixed visual gap.
    X_DODGE = {"FSQ-CE": 0.96, "VQ-CE": 1.04, "DDCL-Cos": 1.00}

    fig, ax = plt.subplots(figsize=(FULL_W * 0.55, 2.6))

    for env_key in ENV_KEYS:
        is_mw = "mw" in env_key
        nom   = nom_bits_map.get(env_key, 2378)
        mkr   = ENV_MARKERS[env_key]

        env_pts_x, env_pts_y, env_pts_col = [], [], []

        for method in TRANSFER_METHODS:
            data      = groups.get((env_key, method), {})
            perf_vals = data.get("suc" if is_mw else "nr", [])
            if not perf_vals:
                continue

            st_y   = compute_stats(perf_vals)
            pm     = st_y["mean"]
            lo_y, hi_y = clip_ci(pm, st_y["ci95"])
            col    = TRANSFER_COLORS[method]
            dodge  = X_DODGE[method]

            if method == "DDCL-Cos":
                bit_vals = data.get("alloc_bits", [])
                if bit_vals:
                    st_x   = compute_stats(bit_vals)
                    bm     = st_x["mean"] * dodge
                    lo_x, hi_x = clip_ci(st_x["mean"], st_x["ci95"], lo=0, hi=np.inf)
                else:
                    bm, lo_x, hi_x = nom * dodge, np.nan, np.nan
            else:
                bm, lo_x, hi_x = nom * dodge, np.nan, np.nan

            ax.errorbar(
                bm, pm,
                xerr=([[lo_x], [hi_x]] if not (np.isnan(lo_x) and np.isnan(hi_x)) else None),
                yerr=([[lo_y], [hi_y]] if not (np.isnan(lo_y) and np.isnan(hi_y)) else None),
                fmt=mkr, color=col, markersize=6, markeredgewidth=0,
                elinewidth=0.7, capsize=2.0, capthick=0.7,
                ecolor=col, alpha=0.9, zorder=3,
            )

            env_pts_x.append(bm)
            env_pts_y.append(pm)
            env_pts_col.append(col)

        # Highlight Pareto-optimal points: same size, black edge ring only
        if len(env_pts_x) >= 2:
            frontier = _pareto_mask(env_pts_x, env_pts_y)
            for xi, yi, col, on_f in zip(env_pts_x, env_pts_y, env_pts_col, frontier):
                if on_f:
                    ax.scatter(xi, yi, marker=mkr, color=col, s=50,
                               edgecolors="black", linewidths=1.4, zorder=5)

    # Nominal budget reference lines
    for nom_val in sorted(set(nom_bits_map.values())):
        ax.axvline(nom_val, color="#BBBBBB", lw=0.6, ls=":", alpha=0.6, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Allocated bits / transition (log scale)")
    ax.set_ylabel("Normalised return / success")
    ax.set_ylim(0, 1.12)

    # Combined legend: methods (colour) + environments (shape)
    method_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=TRANSFER_COLORS[m], markeredgecolor="none",
                   markersize=7, linestyle="None", label=m)
        for m in TRANSFER_METHODS
    ]
    env_handles = [
        plt.Line2D([0], [0], marker=ENV_MARKERS[e], color="w",
                   markerfacecolor="#888888", markeredgecolor="none",
                   markersize=6, linestyle="None", label=ENV_SHORT[e])
        for e in ENV_KEYS
    ]
    frontier_handle = plt.Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="#888888", markeredgecolor="black",
        markersize=6, markeredgewidth=1.4, linestyle="None", label="Per-env. Pareto-optimal",
    )

    leg1 = ax.legend(handles=method_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 1.00), fontsize=FONT_PT - 1,
                     title="Method", title_fontsize=FONT_PT - 1, borderpad=0.3)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=env_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 0.62), fontsize=FONT_PT - 1,
                     title="Env", title_fontsize=FONT_PT - 1, borderpad=0.3)
    ax.add_artist(leg2)
    ax.legend(handles=[frontier_handle], loc="upper left",
              bbox_to_anchor=(1.02, 0.10), fontsize=FONT_PT - 1, borderpad=0.3)

    fig.tight_layout(pad=0.6)
    savefig(outdir, "07_transfer_rate_efficiency")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 08: Transfer codebook usage
# ═══════════════════════════════════════════════════════════════════════════════

def fig08_transfer_codebook_usage(groups: dict, outdir: Path) -> None:
    """Grouped bar: codebook usage % per env, colour = method."""
    n_e   = len(ENV_KEYS)
    n_m   = len(TRANSFER_METHODS)
    grp_w = 0.80
    bar_w = grp_w / n_m
    x_base = np.arange(n_e)

    fig, ax = plt.subplots(figsize=(FULL_W * 0.65, 2.0))

    for mi, method in enumerate(TRANSFER_METHODS):
        offset = (mi - (n_m - 1) / 2) * bar_w
        col    = TRANSFER_COLORS[method]
        xs, means, lo_errs, hi_errs = [], [], [], []

        for ei, env_key in enumerate(ENV_KEYS):
            vals = groups.get((env_key, method), {}).get("usage_pct", [])
            if not vals:
                continue
            st   = compute_stats(vals)
            mean = st["mean"]
            lo_e, hi_e = clip_ci(mean, st["ci95"], lo=0, hi=100)
            xs.append(x_base[ei] + offset)
            means.append(mean)
            lo_errs.append(lo_e)
            hi_errs.append(hi_e)

        if not xs:
            continue
        ax.bar(xs, means, width=bar_w * 0.9, color=col, label=method, zorder=2)
        ax.errorbar(
            xs, means, yerr=[lo_errs, hi_errs],
            fmt="none", ecolor="#333333",
            capsize=2, capthick=0.8, elinewidth=0.8, zorder=3,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(ENV_LABELS, fontsize=FONT_PT - 1)
    ax.set_ylabel("Codebook usage (%)")
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=FONT_PT - 1)

    savefig(outdir, "08_transfer_codebook_usage")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="Output directory for PNG/PDF figures.")
    parser.add_argument("--toy-only", action="store_true",
                        help="Generate only toy figures (no W&B needed).")
    parser.add_argument("--transfer-only", action="store_true",
                        help="Generate only transfer figures (requires W&B).")
    parser.add_argument("--no-stoch", action="store_true",
                        help="Skip Figure 05 (stochasticity ablation).")
    args = parser.parse_args()

    apply_paper_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.outdir}\n")

    do_toy      = not args.transfer_only
    do_transfer = not args.toy_only

    # ── Toy figures ───────────────────────────────────────────────────────────
    if do_toy:
        print("Loading toy data from canonical JSON files …")
        toy_data = load_toy_data()

        print("\nGenerating Figure 03: toy rate-precision Pareto …")
        fig03_toy_pareto(toy_data, args.outdir)

        print("Generating Figure 04: DDCL objective ablation …")
        fig04_objective_ablation(toy_data, args.outdir)

        if not args.no_stoch:
            print("Generating Figure 05: stochasticity ablation …")
            stoch_data = load_stoch_ablation_data()
            fig05_stoch_ablation(stoch_data, args.outdir)

    # ── Transfer figures ──────────────────────────────────────────────────────
    if do_transfer:
        print("\nFetching transfer runs from W&B …")
        all_records = fetch_transfer_runs()
        if not all_records:
            print("  No transfer records found; skipping transfer figures.")
        else:
            selected = select_best_ddcl_lambda(all_records)
            groups   = group_transfer_records(selected)

            print("\nCoverage check (env × method):")
            for env in ENV_KEYS:
                for m in TRANSFER_METHODS:
                    n    = len(groups.get((env, m), {}).get("nr", []) +
                               groups.get((env, m), {}).get("suc", []))
                    flag = "✅" if n >= 5 else f"⚠  {n}/5"
                    print(f"  {env:<22} {m:<12} {flag}")

            print("\nGenerating Figure 06: transfer performance …")
            fig06_transfer_performance(groups, args.outdir)

            print("Generating Figure 07: transfer rate efficiency …")
            fig07_transfer_rate_efficiency(groups, all_records, args.outdir)

            print("Generating Figure 08: transfer codebook usage …")
            fig08_transfer_codebook_usage(groups, args.outdir)

    print(f"\nDone. All figures written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
