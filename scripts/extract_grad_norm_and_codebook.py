#!/usr/bin/env python3
"""Extract grad-norm (E2) and codebook-stability (E5) data from W&B.

E2 — Gradient-norm comparison (C5/T5):
  Pulls train/grad_norm (world-model encoder) and q_grad_norm (Q-network),
  pi_grad_norm (policy) for DDCL-CE, DDCL-Cos, FSQ-CE, VQ-CE (5 seeds each).
  Expected per T5: DDCL ≈ FSQ (bounded tanh-STE Jacobian), VQ higher/more
  variable (unbounded nearest-neighbor + identity STE).

E5 — Codebook-stability curves (C6/T6a):
  Pulls eval/codebook/usage_percent and eval/codebook/per_group_entropy_mean
  over training steps for DDCL and VQ. Shows DDCL's implicit fixed grid never
  collapses (structurally: any |z| can reach any bin), while VQ may show
  declining usage from dead-vector collapse.

Data source: W&B project ddcl_mbrl_toy, final-v1 training runs.

Output:
  private/Metrics/Toy/Corrected Reevaluation/e2_grad_norm_<date>.json
  private/Metrics/Toy/Corrected Reevaluation/e5_codebook_stability_<date>.json

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/extract_grad_norm_and_codebook.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb

_ROOT = Path(__file__).parent.parent
os.chdir(_ROOT)

WANDB_PROJECT = "ddcl_mbrl_toy"
OUT_DIR = Path("private/Metrics/Toy/Corrected Reevaluation")

# ── Method → run name pattern (final-v1 canonical runs) ───────────────────────
METHODS = {
    "ddcl_ce":      "ddcl-cross-entropy-stable-final-v1",
    "ddcl_cosine":  "ddcl-cosine-stable-final-v1",
    "fsq_ce":       "fsq-cross-entropy-stable-final-v1",
    "vq_ce":        "vq-cross-entropy-stable-final-v1",
}
METHOD_LABELS = {
    "ddcl_ce":     "DDCL-CE",
    "ddcl_cosine": "DDCL-Cos",
    "fsq_ce":      "FSQ-CE",
    "vq_ce":       "VQ-CE",
}
N_SEEDS = 5

# ── History keys ───────────────────────────────────────────────────────────────
GRAD_KEY       = "grad_norm"          # WM encoder grad norm
Q_GRAD_KEY     = "q_grad_norm"        # Q-network grad norm
PI_GRAD_KEY    = "pi_grad_norm"       # Policy grad norm
USAGE_KEY      = "eval/.codebook/usage_percent"
ENTROPY_KEY    = "eval/.codebook/per_group_entropy_mean"
STEP_KEY       = "_step"

N_HISTORY_SAMPLES = 100  # points to sample along the training trajectory


def fetch_run_history(
    api: wandb.Api,
    run_name_pattern: str,
    keys: list[str],
    n_seeds: int = 5,
    n_samples: int = 100,
) -> list[dict]:
    """Fetch history for all seeds matching a run name pattern.

    Returns a list of dicts, one per seed, each containing:
      {seed_idx, run_id, run_name, history: {key: [values]}}
    """
    print(f"  Fetching '{run_name_pattern}' seeds 0–{n_seeds-1} ...")

    # Find matching runs
    all_runs = list(api.runs(
        WANDB_PROJECT,
        filters={"display_name": {"$regex": f".*{run_name_pattern}.*"}},
        per_page=50,
    ))
    print(f"    Found {len(all_runs)} matching runs.")

    seed_data = []
    for run in sorted(all_runs, key=lambda r: (r.name or "")):
        if run.state not in ("finished", "crashed"):
            print(f"    Skip {run.name} (state={run.state})")
            continue
        if len(seed_data) >= n_seeds:
            break

        try:
            # Do NOT pass keys= — it causes history to return empty for some W&B versions.
            # Fetch all columns and filter after download.
            hist_df = run.history(samples=n_samples)
            if hist_df.empty:
                # Fall back to summary for scalar stats
                summary = dict(run.summary or {})
                entry = {
                    "seed_idx": len(seed_data),
                    "run_id":   run.id,
                    "run_name": run.name,
                    "history":  {},
                    "summary":  {},
                }
                for k in keys:
                    clean_k = k.replace("eval/.", "eval/").replace("train/.", "train/")
                    if k in summary:
                        entry["summary"][k] = float(summary[k])
                    elif clean_k in summary:
                        entry["summary"][clean_k] = float(summary[clean_k])
                seed_data.append(entry)
                print(f"    {run.name}: history empty, using summary")
                continue

            entry = {
                "seed_idx": len(seed_data),
                "run_id":   run.id,
                "run_name": run.name,
                "history":  {},
                "summary":  {},
            }
            for k in keys + [STEP_KEY]:
                if k in hist_df.columns:
                    vals = hist_df[k].dropna().tolist()
                    entry["history"][k] = [float(v) for v in vals]

            # Also get summary for single-number stats
            summary = dict(run.summary or {})
            for k in keys:
                clean_k = k.replace("eval/.", "eval/").replace("train/.", "train/")
                for sk in [k, clean_k, k.lstrip(".")]:
                    if sk in summary:
                        try:
                            entry["summary"][k] = float(summary[sk])
                        except (TypeError, ValueError):
                            pass
                        break

            n_grad = len(entry["history"].get(GRAD_KEY, []))
            n_cb   = len(entry["history"].get(USAGE_KEY, []))
            print(f"    {run.name}: grad_norm={n_grad} pts, usage={n_cb} pts")
            seed_data.append(entry)

        except Exception as e:
            print(f"    Error for {run.name}: {e}")

    return seed_data


def aggregate_grad_norm(seed_data: list[dict]) -> dict:
    """Compute mean/std/median of grad_norm summary stats across seeds."""
    results = {}
    for grad_key, label in [
        (GRAD_KEY,    "encoder"),
        (Q_GRAD_KEY,  "Q-network"),
        (PI_GRAD_KEY, "policy"),
    ]:
        vals = []
        for entry in seed_data:
            # Prefer the time-averaged value from history; fall back to summary
            if grad_key in entry["history"] and entry["history"][grad_key]:
                vals.append(float(np.mean(entry["history"][grad_key])))
            elif grad_key in entry["summary"]:
                vals.append(entry["summary"][grad_key])
        if vals:
            arr = np.array(vals)
            results[label] = {
                "mean":   float(arr.mean()),
                "std":    float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
                "median": float(np.median(arr)),
                "values": [float(v) for v in arr],
                "n":      len(arr),
            }
    return results


def aggregate_codebook_stability(seed_data: list[dict]) -> dict:
    """Aggregate codebook usage and entropy over training."""
    results = {}

    for key, label in [(USAGE_KEY, "usage_percent"), (ENTROPY_KEY, "entropy_mean")]:
        all_histories = []
        final_vals = []
        for entry in seed_data:
            hist = entry["history"].get(key, [])
            if hist:
                all_histories.append(hist)
                final_vals.append(hist[-1])
            elif key in entry["summary"]:
                final_vals.append(entry["summary"][key])

        if not final_vals:
            continue

        arr_final = np.array(final_vals)
        results[label] = {
            "final_mean":   float(arr_final.mean()),
            "final_std":    float(arr_final.std(ddof=1)) if len(arr_final) > 1 else float("nan"),
            "final_values": [float(v) for v in arr_final],
            "n":            len(arr_final),
            # Per-seed training trajectories (for stability curves)
            "trajectories": [[float(v) for v in h] for h in all_histories],
        }
    return results


def main() -> int:
    api = wandb.Api(timeout=60)

    grad_all: dict[str, list[dict]] = {}
    cb_all:   dict[str, list[dict]] = {}

    grad_keys  = [GRAD_KEY, Q_GRAD_KEY, PI_GRAD_KEY]
    cb_keys    = [USAGE_KEY, ENTROPY_KEY]

    for method_key, pattern in METHODS.items():
        print(f"\n── Method: {METHOD_LABELS[method_key]} ({method_key}) ──────────────")
        seed_data = fetch_run_history(
            api, pattern, keys=grad_keys + cb_keys, n_seeds=N_SEEDS,
            n_samples=N_HISTORY_SAMPLES,
        )
        grad_all[method_key] = seed_data
        cb_all[method_key]   = seed_data

    # ── E2: Gradient-norm output ─────────────────────────────────────────────
    print("\n── E2: Gradient-norm summary ───────────────────────────────────────")
    print(f"  {'Method':<14} {'Encoder μ':>10}  {'Q-net μ':>10}  {'Policy μ':>10}  n")
    e2_output: dict = {}
    for method_key, seed_data in grad_all.items():
        agg = aggregate_grad_norm(seed_data)
        e2_output[method_key] = {
            "label": METHOD_LABELS[method_key],
            "grad_norm": agg,
            "seed_runs": [{"run_id": s["run_id"], "run_name": s["run_name"]}
                          for s in seed_data],
        }
        enc_mean   = agg.get("encoder",   {}).get("mean",   float("nan"))
        q_mean     = agg.get("Q-network", {}).get("mean",   float("nan"))
        pi_mean    = agg.get("policy",    {}).get("mean",   float("nan"))
        n          = agg.get("encoder",   {}).get("n",      0)
        print(f"  {METHOD_LABELS[method_key]:<14} {enc_mean:>10.4f}  {q_mean:>10.4f}  {pi_mean:>10.4f}  {n}")

    # ── E5: Codebook-stability output ────────────────────────────────────────
    print("\n── E5: Codebook-stability summary ──────────────────────────────────")
    print(f"  {'Method':<14} {'Usage % (final)':>18}  {'H_group (final)':>18}  n")
    e5_output: dict = {}
    for method_key, seed_data in cb_all.items():
        agg = aggregate_codebook_stability(seed_data)
        e5_output[method_key] = {
            "label": METHOD_LABELS[method_key],
            "codebook_stability": agg,
            "seed_runs": [{"run_id": s["run_id"], "run_name": s["run_name"]}
                          for s in seed_data],
        }
        usage  = agg.get("usage_percent", {})
        entropy = agg.get("entropy_mean", {})
        u_mean = usage.get("final_mean",  float("nan"))
        u_std  = usage.get("final_std",   float("nan"))
        e_mean = entropy.get("final_mean", float("nan"))
        n      = usage.get("n", 0)
        print(f"  {METHOD_LABELS[method_key]:<14} {u_mean:>7.2f}±{u_std:>6.2f}%  "
              f"{e_mean:>12.4f}  {n}")
        # Trajectory info
        trajs = usage.get("trajectories", [])
        if trajs:
            print(f"    trajectory samples: {len(trajs)} seeds, "
                  f"avg len={int(np.mean([len(t) for t in trajs]))} pts, "
                  f"first pt={np.mean([t[0] for t in trajs]):.2f}%, "
                  f"last pt={np.mean([t[-1] for t in trajs]):.2f}%")

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    e2_path = OUT_DIR / f"e2_grad_norm_{date_str}.json"
    with open(e2_path, "w") as f:
        json.dump(e2_output, f, indent=2)
    print(f"\nSaved E2: {e2_path}")

    e5_path = OUT_DIR / f"e5_codebook_stability_{date_str}.json"
    with open(e5_path, "w") as f:
        json.dump(e5_output, f, indent=2)
    print(f"Saved E5: {e5_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
