#!/usr/bin/env python3
"""Generate paper metric plots from W&B or local CSV exports.

Expected CSV columns are the metric names logged by train.py, optionally with
configuration columns such as env, agent, seed, agent.ddcl_lambda, and
agent.ddcl_delta. The script is intentionally permissive so it can consume raw
W&B exports after light column renaming.

All figures are sized and styled for a 10 pt two-column workshop paper (RLC 2026).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

tmp_dir = Path(os.getenv("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_dir / "xdg-cache"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ── CRITICAL: rate axis selection ─────────────────────────────────────────────
# For CROSS-METHOD comparisons (DDCL vs FSQ vs VQ), use the shared-dataset
# realized rate from scripts/measure_shared_dataset_rate.py (E7).
# Own-policy empirical entropy is CONFOUNDED by state-visitation (App. T6/C7)
# and must NOT be used as a cross-method x-axis.
#
# For DDCL-ONLY ablations (λ/scale sweep within one method), the own-policy
# empirical entropy or allocated_bits are acceptable within-method axes.
#
# DEFAULT_RATE is kept as the own-policy key for DDCL-internal diagnostics.
# Cross-method Pareto plots route through gen_pareto_fig.py (E7 realized rate).
DEFAULT_RATE = "rate/empirical_entropy_bits_per_transition"  # own-policy diagnostic ONLY
DEFAULT_RETURN = "episodic_return"
DEFAULT_SUCCESS = "episodic_success"
DEFAULT_PRECISION_SUCCESS = "episodic_precision_success_0p02"

# ── Paper-layout constants (RLC 10 pt, 2-column) ──────────────────────────────
COL_W = 3.25    # single-column width, inches
FULL_W = 6.75   # full-page width, inches
FONT_SIZE = 8   # body text, pt


def apply_paper_style() -> None:
    """Apply rcParams suitable for a 10 pt two-column workshop paper."""
    plt.rcParams.update({
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "pdf.fonttype":        42,   # TrueType (not Type 3) — required by most venues
        "ps.fonttype":         42,
        "font.size":           FONT_SIZE,
        "axes.titlesize":      FONT_SIZE,
        "axes.labelsize":      FONT_SIZE,
        "xtick.labelsize":     FONT_SIZE - 1,
        "ytick.labelsize":     FONT_SIZE - 1,
        "legend.fontsize":     FONT_SIZE - 1,
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


def _savefig(outdir: Path, stem: str) -> None:
    """Save PNG (300 dpi) and PDF (vector, TrueType fonts) then close figure."""
    plt.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close()


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        clean = col
        for prefix in ("eval/", "rollout/", "train/"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        rename[col] = clean
    return df.rename(columns=rename)


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_pareto(df: pd.DataFrame, outdir: Path, y: str) -> None:
    """Pareto scatter (empirical entropy bits vs. *y*).

    x-axis: empirical entropy bits/transition (cross-method fair metric).
    errorbar: sd (consistent with make_paper_plots.py).
    Outputs: PNG + PDF.
    """
    if DEFAULT_RATE not in df or y not in df:
        return
    hue = "agent" if "agent" in df else None
    style = "env" if "env" in df else None
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    sns.scatterplot(
        data=df, x=DEFAULT_RATE, y=y, hue=hue, style=style,
        s=22, alpha=0.5, linewidth=0, ax=ax,
    )
    sns.lineplot(
        data=df,
        x=DEFAULT_RATE,
        y=y,
        hue=hue,
        estimator="mean",
        errorbar="sd",
        legend=False,
        alpha=0.7,
        ax=ax,
    )
    ax.set_xlabel("Empirical entropy bits / transition")
    ax.set_ylabel("Success rate" if "success" in y else "Episode return")
    _savefig(outdir, f"pareto_{y.replace('/', '_')}")


def plot_rate_sensitivity(df: pd.DataFrame, outdir: Path, y: str) -> None:
    """Line plot of success/return vs. DDCL rate-penalty lambda."""
    lambda_cols = [c for c in ("agent.ddcl_lambda", "ddcl_lambda", "lambda") if c in df]
    if not lambda_cols or y not in df:
        return
    lambda_col = lambda_cols[0]
    hue = "env" if "env" in df else None
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    sns.lineplot(
        data=df,
        x=lambda_col,
        y=y,
        hue=hue,
        marker="o",
        estimator="mean",
        errorbar="sd",
        ax=ax,
    )
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xlabel(r"DDCL rate penalty $\lambda$")
    ax.set_ylabel("Success rate" if "success" in y else "Episode return")
    _savefig(outdir, f"lambda_sensitivity_{y.replace('/', '_')}")


def plot_code_usage(df: pd.DataFrame, outdir: Path) -> None:
    """Scatter of normalized code entropy vs. empirical entropy bits."""
    x = "rate/empirical_entropy_bits_per_transition"
    y = "codebook/per_group_entropy_mean"
    if x not in df or y not in df:
        return
    hue = "agent" if "agent" in df else None
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, s=28, linewidth=0, ax=ax)
    ax.set_xlabel("Empirical entropy bits / transition")
    ax.set_ylabel("Normalized code entropy")
    _savefig(outdir, "code_entropy_vs_rate")


def plot_horizon_diagnostics(df: pd.DataFrame, outdir: Path) -> None:
    """Per-horizon loss breakdown (reward and TC loss across imagined steps)."""
    horizon_cols = [
        c for c in df.columns
        if c.startswith("wm/reward_loss_h") or c.startswith("wm/tc_loss_h")
    ]
    if not horizon_cols:
        return
    id_cols = [c for c in ("env", "agent", "seed", "step", "env_step") if c in df]
    long = df.melt(
        id_vars=id_cols,
        value_vars=horizon_cols,
        var_name="metric",
        value_name="value",
    )
    long["horizon"] = long["metric"].str.extract(r"h(\d+)").astype(int)
    long["loss"] = long["metric"].str.replace(r"_h\d+$", "", regex=True)
    hue = "agent" if "agent" in long else None
    for loss_name, loss_df in long.groupby("loss"):
        fig, ax = plt.subplots(figsize=(COL_W, 2.2))
        sns.lineplot(
            data=loss_df,
            x="horizon",
            y="value",
            hue=hue,
            marker="o",
            estimator="mean",
            errorbar="sd",
            ax=ax,
        )
        ax.set_xlabel("Rollout horizon")
        ax.set_ylabel(loss_name)
        safe_name = loss_name.replace("/", "_")
        _savefig(outdir, f"horizon_{safe_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-quality metric plots from DDCL-MBRL CSV exports."
    )
    parser.add_argument("csv", type=Path, nargs="+", help="Input metric CSV file(s).")
    parser.add_argument("--outdir", type=Path, default=Path("paper_metric_plots"))
    parser.add_argument(
        "--score",
        choices=(DEFAULT_RETURN, DEFAULT_SUCCESS, DEFAULT_PRECISION_SUCCESS),
        default=DEFAULT_RETURN,
        help="Primary y-axis score for Pareto and sensitivity plots.",
    )
    args = parser.parse_args()

    apply_paper_style()          # must come before any plt calls
    _ensure_outdir(args.outdir)
    df = pd.concat(
        (_clean_columns(pd.read_csv(path)) for path in args.csv), ignore_index=True
    )

    plot_pareto(df, args.outdir, args.score)
    plot_rate_sensitivity(df, args.outdir, args.score)
    plot_code_usage(df, args.outdir)
    plot_horizon_diagnostics(df, args.outdir)


if __name__ == "__main__":
    main()
