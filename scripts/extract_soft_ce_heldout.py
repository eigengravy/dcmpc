#!/usr/bin/env python3
"""Extract DDCL Soft CE heldout evaluation results from W&B and save to JSON.

Fetches the 5 runs tagged 'paper-heldout-soft_ce_heldout--*' from the
ddcl_mbrl_toy_eval W&B project, aggregates metrics across seeds, and saves
the result to the canonical heldout eval directory in the format expected by
gen_objective_ablation_fig.py and gen_pareto_fig.py.

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/extract_soft_ce_heldout.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb

# ── Config ────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "ddcl_mbrl_toy_eval"
RUN_PREFIX    = "paper-heldout-soft_ce_heldout--toy-precision-gate-final--ddcl_soft_ce"
OUT_DIR       = Path("private/Metrics/Toy/Corrected Reevaluation")

METRICS_OF_INTEREST = [
    "episodic_success",
    "episodic_precision_success_0p02",
    "episodic_precision_success_0p04",
    "episodic_precision_success_0p08",
    "episodic_return",
    "rate/empirical_entropy_bits_per_transition",
    "rate/allocated_bits_per_transition",   # may be inflated pre-2026-05-21; corrected below
    "rate/ddcl_loss_bits_per_dim",          # for unsigned-formula correction
    "rate/ddcl_signed_bits_per_dim",        # denominator for n_total_dims derivation
    "rate/ddcl_signed_bits_per_transition", # numerator for n_total_dims derivation
    "codebook/usage_percent",
    "codebook/per_group_entropy_mean",
]

def main() -> int:
    api = wandb.Api()
    print(f"Fetching runs from {WANDB_PROJECT} matching '{RUN_PREFIX}--s*' ...")
    runs = list(api.runs(
        path=WANDB_PROJECT,
        filters={"display_name": {"$regex": f"^{RUN_PREFIX}--s"}},
        per_page=20,
    ))
    print(f"  Found {len(runs)} matching runs.")
    if not runs:
        # Try broader filter
        runs = list(api.runs(path=WANDB_PROJECT, per_page=100))
        runs = [r for r in runs if RUN_PREFIX in (r.name or "")]
        print(f"  Fallback broad filter: {len(runs)} runs.")

    if not runs:
        print("ERROR: No runs found. Check W&B project and run names.")
        return 1

    # Collect per-run metrics
    run_ids   = []
    metric_values: dict[str, list[float]] = {m: [] for m in METRICS_OF_INTEREST}

    for run in sorted(runs, key=lambda r: r.name or ""):
        if run.state not in ("finished", "crashed"):
            print(f"  Skipping run {run.name} (state={run.state})")
            continue
        summary = dict(run.summary or {})
        # eval.py logs metrics under an "eval/" SummarySubDict.
        # SummarySubDict is NOT a subclass of dict, so use hasattr check.
        eval_ns_raw = summary.get("eval/")
        eval_ns = eval_ns_raw if (eval_ns_raw is not None and hasattr(eval_ns_raw, "get")) else {}
        run_ids.append(run.id)
        print(f"  Run {run.name} (id={run.id}, state={run.state})")
        for metric in METRICS_OF_INTEREST:
            # Use explicit None checks — do NOT use `or` since 0.0 is falsy.
            val = eval_ns.get(metric)
            if val is None:
                val = summary.get(metric)
            if val is None:
                val = summary.get(metric.replace("/", "."))
            if val is not None:
                try:
                    metric_values[metric].append(float(val))
                    print(f"    {metric}: {float(val):.4f}")
                except (TypeError, ValueError):
                    pass
            else:
                print(f"    {metric}: MISSING")

    if not run_ids:
        print("ERROR: No finished/crashed runs found.")
        return 1

    # ── Correct DDCL allocated_bits to use the unsigned formula ───────────────
    # DDCL runs before 2026-05-21 logged rate/allocated_bits_per_transition with
    # the signed formula log2(2|z|/δ+1), which inflates the value by ~36% vs
    # the DDCL-paper unsigned formula log2(|z|/δ+1) used in training comm_loss.
    # Corrected = rate/ddcl_loss_bits_per_dim × n_total_dims, where
    # n_total_dims = ddcl_signed_bits_per_transition / ddcl_signed_bits_per_dim.
    loss_pd_vals  = metric_values.get("rate/ddcl_loss_bits_per_dim", [])
    signed_pd_vals = metric_values.get("rate/ddcl_signed_bits_per_dim", [])
    signed_tot_vals = metric_values.get("rate/ddcl_signed_bits_per_transition", [])

    if (len(loss_pd_vals) == len(signed_pd_vals) == len(signed_tot_vals) > 0
            and all(abs(v) > 1e-9 for v in signed_pd_vals)):
        corrected_vals = []
        for loss, spd, stot in zip(loss_pd_vals, signed_pd_vals, signed_tot_vals):
            n_total_dims = round(stot / spd)
            corrected_vals.append(loss * n_total_dims)
        raw_vals = metric_values.get("rate/allocated_bits_per_transition", [])
        if raw_vals:
            raw_mean = float(np.mean(raw_vals))
            corr_mean = float(np.mean(corrected_vals))
            print(f"\n  Rate correction: raw allocated_bits mean={raw_mean:.2f} → "
                  f"corrected={corr_mean:.2f} (unsigned formula, Δ={corr_mean - raw_mean:.2f})")
        metric_values["rate/allocated_bits_per_transition"] = corrected_vals
        print(f"  Using corrected allocated_bits (unsigned formula log2(|z|/δ+1) × n_total_dims)")
    elif metric_values.get("rate/allocated_bits_per_transition"):
        print("  WARNING: Could not compute rate correction (missing ddcl_signed metrics). "
              "Using raw allocated_bits_per_transition — may be inflated for pre-2026-05-21 runs.")

    # Build output dict in the canonical heldout format
    # Drop the auxiliary correction metrics — they are not needed in the output JSON
    _internal_keys = {"rate/ddcl_loss_bits_per_dim", "rate/ddcl_signed_bits_per_dim",
                      "rate/ddcl_signed_bits_per_transition"}
    metrics_blob: dict = {}
    for metric, values in metric_values.items():
        if metric in _internal_keys or not values:
            continue
        arr = np.array(values, dtype=float)
        metrics_blob[metric] = {
            "n":      len(arr),
            "mean":   float(arr.mean()),
            "std":    float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
            "values": [float(v) for v in arr],
        }

    output = {
        "ddcl_soft_ce": {
            "ids":        run_ids,
            "name_count": len(run_ids),
            "metrics":    metrics_blob,
        }
    }

    # Print summary
    print("\n── DDCL Soft CE heldout summary ─────────────────────────────────")
    for m in ["episodic_success", "episodic_precision_success_0p02",
              "episodic_precision_success_0p04", "episodic_return"]:
        if m in metrics_blob:
            d = metrics_blob[m]
            print(f"  {m}: {d['mean']:.3f} ± {d['std']:.3f}  (n={d['n']})")
            print(f"    per-seed: {[f'{v:.3f}' for v in d['values']]}")

    # Save JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = OUT_DIR / f"wandb_toy_soft_ce_5seed_heldout_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
