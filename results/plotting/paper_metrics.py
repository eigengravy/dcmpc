#!/usr/bin/env python3
"""Generate paper metric plots from W&B or local CSV exports.

Expected CSV columns are the metric names logged by train.py, optionally with
configuration columns such as env, agent, seed, agent.ddcl_lambda, and
agent.ddcl_delta. The script is intentionally permissive so it can consume raw
W&B exports after light column renaming.
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


DEFAULT_RATE = "rate/allocated_bits_per_transition"
DEFAULT_RETURN = "episodic_return"
DEFAULT_SUCCESS = "episodic_success"
DEFAULT_PRECISION_SUCCESS = "episodic_precision_success_0p02"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        clean = col
        for prefix in ("eval/", "rollout/", "train/"):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        rename[col] = clean
    return df.rename(columns=rename)


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_pareto(df: pd.DataFrame, outdir: Path, y: str) -> None:
    if DEFAULT_RATE not in df or y not in df:
        return
    hue = "agent" if "agent" in df else None
    style = "env" if "env" in df else None
    plt.figure(figsize=(7.0, 4.8))
    sns.scatterplot(data=df, x=DEFAULT_RATE, y=y, hue=hue, style=style, s=72)
    sns.lineplot(
        data=df,
        x=DEFAULT_RATE,
        y=y,
        hue=hue,
        estimator="mean",
        errorbar=("ci", 95),
        legend=False,
        alpha=0.45,
    )
    plt.xlabel("Allocated bits / transition")
    plt.ylabel("Success rate" if "success" in y else "Episode return")
    plt.tight_layout()
    plt.savefig(outdir / f"pareto_{y.replace('/', '_')}.png", dpi=300)
    plt.close()


def plot_rate_sensitivity(df: pd.DataFrame, outdir: Path, y: str) -> None:
    lambda_cols = [c for c in ("agent.ddcl_lambda", "ddcl_lambda", "lambda") if c in df]
    if not lambda_cols or y not in df:
        return
    lambda_col = lambda_cols[0]
    hue = "env" if "env" in df else None
    plt.figure(figsize=(7.0, 4.8))
    sns.lineplot(
        data=df,
        x=lambda_col,
        y=y,
        hue=hue,
        marker="o",
        estimator="mean",
        errorbar=("ci", 95),
    )
    plt.xscale("symlog", linthresh=1e-6)
    plt.xlabel("DDCL rate penalty")
    plt.ylabel("Success rate" if "success" in y else "Episode return")
    plt.tight_layout()
    plt.savefig(outdir / f"lambda_sensitivity_{y.replace('/', '_')}.png", dpi=300)
    plt.close()


def plot_code_usage(df: pd.DataFrame, outdir: Path) -> None:
    x = "rate/allocated_bits_per_transition"
    y = "codebook/per_group_entropy_mean"
    if x not in df or y not in df:
        return
    hue = "agent" if "agent" in df else None
    plt.figure(figsize=(7.0, 4.8))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, s=72)
    plt.xlabel("Allocated bits / transition")
    plt.ylabel("Normalized code entropy")
    plt.tight_layout()
    plt.savefig(outdir / "code_entropy_vs_rate.png", dpi=300)
    plt.close()


def plot_horizon_diagnostics(df: pd.DataFrame, outdir: Path) -> None:
    horizon_cols = [
        c
        for c in df.columns
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
        plt.figure(figsize=(7.0, 4.8))
        sns.lineplot(
            data=loss_df,
            x="horizon",
            y="value",
            hue=hue,
            marker="o",
            estimator="mean",
            errorbar=("ci", 95),
        )
        plt.xlabel("Rollout horizon")
        plt.ylabel(loss_name)
        plt.tight_layout()
        safe_name = loss_name.replace("/", "_")
        plt.savefig(outdir / f"horizon_{safe_name}.png", dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="+", help="Input metric CSV file(s).")
    parser.add_argument("--outdir", type=Path, default=Path("paper_metric_plots"))
    parser.add_argument(
        "--score",
        choices=(DEFAULT_RETURN, DEFAULT_SUCCESS, DEFAULT_PRECISION_SUCCESS),
        default=DEFAULT_RETURN,
        help="Primary y-axis score for Pareto and sensitivity plots.",
    )
    args = parser.parse_args()

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
