#!/usr/bin/env python3
"""Extract DDCL stochasticity-ablation heldout results from W&B → JSON.

Fetches the 4 stochastic conditions × 5 seeds from W&B project
ddcl_mbrl_toy_eval (run names: paper-heldout-stoch_ablation--*), applies the
DDCL rate correction (unsigned formula, 2026-05-21), and saves the canonical
stochasticity-ablation JSON used by gen_stoch_ablation_fig.py.

The deterministic-eval / deterministic-targets (d,d) condition is read from the
existing canonical heldout JSON (wandb_toy_all_methods_5seed_heldout_*.json)
under the key 'ddcl_ce', which already has 5 seeds.

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/extract_stoch_ablation_heldout.py
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

# Stochastic conditions: (label, run_name_infix, eval_mode, target_mode)
STOCH_CONDITIONS = [
    ("ddcl_ce_ss",   "ddcl_ce_stoch",      "stoch", "stoch"),   # (s,s)
    ("ddcl_ce_sd",   "ddcl_ce_stoch_eval", "stoch", "det"),     # (s,d)
    ("ddcl_ce_ds",   "ddcl_ce_stoch_tgt",  "det",   "stoch"),   # (d,s)
    ("ddcl_cos_sd",  "ddcl_cos_stoch_eval","stoch", "det"),     # Cos (s,d)
]

# The (d,d) condition comes from the canonical 5-seed JSON
CANONICAL_HELDOUT_GLOB = "private/Metrics/Toy/Corrected Reevaluation/wandb_toy_all_methods_5seed_heldout_*.json"
CANONICAL_CE_KEY = "ddcl_ce"

OUT_DIR = Path("private/Metrics/Toy/Corrected Reevaluation")

METRICS_OF_INTEREST = [
    "episodic_success",
    "episodic_precision_success_0p02",
    "episodic_precision_success_0p04",
    "episodic_precision_success_0p08",
    "episodic_return",
    "rate/empirical_entropy_bits_per_transition",
    "rate/allocated_bits_per_transition",
    # DDCL rate correction metrics (unsigned formula, 2026-05-21)
    "rate/ddcl_loss_bits_per_dim",
    "rate/ddcl_signed_bits_per_dim",
    "rate/ddcl_signed_bits_per_transition",
    "codebook/usage_percent",
    "codebook/per_group_entropy_mean",
]

_INTERNAL_KEYS = {
    "rate/ddcl_loss_bits_per_dim",
    "rate/ddcl_signed_bits_per_dim",
    "rate/ddcl_signed_bits_per_transition",
}

RUN_NAME_PREFIX = "paper-heldout-stoch_ablation--toy-precision-gate-final"


def _correct_allocated_bits(metric_values: dict) -> dict:
    """Apply unsigned-formula correction to DDCL allocated_bits."""
    loss_pd    = metric_values.get("rate/ddcl_loss_bits_per_dim", [])
    signed_pd  = metric_values.get("rate/ddcl_signed_bits_per_dim", [])
    signed_tot = metric_values.get("rate/ddcl_signed_bits_per_transition", [])

    if (len(loss_pd) == len(signed_pd) == len(signed_tot) > 0
            and all(abs(v) > 1e-9 for v in signed_pd)):
        corrected = []
        for l, spd, stot in zip(loss_pd, signed_pd, signed_tot):
            n_dims = round(stot / spd)
            corrected.append(l * n_dims)
        raw = metric_values.get("rate/allocated_bits_per_transition", [])
        if raw:
            print(f"    rate correction: raw={np.mean(raw):.2f} → corrected={np.mean(corrected):.2f}")
        metric_values["rate/allocated_bits_per_transition"] = corrected
    elif metric_values.get("rate/allocated_bits_per_transition"):
        print("    WARNING: Cannot compute rate correction; allocated_bits may be inflated.")
    return metric_values


def _aggregate(values_dict: dict) -> dict:
    """Convert {metric: [float]} → {metric: {n, mean, std, values}}."""
    blob = {}
    for metric, values in values_dict.items():
        if metric in _INTERNAL_KEYS or not values:
            continue
        arr = np.array(values, dtype=float)
        blob[metric] = {
            "n":      len(arr),
            "mean":   float(arr.mean()),
            "std":    float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
            "values": [float(v) for v in arr],
        }
    return blob


def fetch_condition(api, infix: str) -> dict:
    """Fetch 5 seeds for a stochastic condition from W&B."""
    name_pattern = f"{RUN_NAME_PREFIX}--{infix}--s"
    print(f"\nFetching condition '{infix}' (prefix='{name_pattern}*') ...")

    runs = list(api.runs(
        path=WANDB_PROJECT,
        filters={"display_name": {"$regex": f"^{name_pattern}"}},
        per_page=20,
    ))
    print(f"  Found {len(runs)} runs.")

    if not runs:
        # Broad fallback
        all_runs = list(api.runs(path=WANDB_PROJECT, per_page=200))
        runs = [r for r in all_runs if name_pattern in (r.name or "")]
        print(f"  Broad fallback: {len(runs)} runs.")

    run_ids = []
    mv: dict[str, list[float]] = {m: [] for m in METRICS_OF_INTEREST}

    for run in sorted(runs, key=lambda r: r.name or ""):
        if run.state not in ("finished", "crashed"):
            print(f"  Skipping {run.name} (state={run.state})")
            continue
        summary = dict(run.summary or {})
        # eval.py logs metrics under an "eval/" SummarySubDict in the W&B summary.
        # Try three lookup strategies:
        #   1. Nested under "eval/" namespace (SummarySubDict) — standard eval.py behaviour
        #   2. Flat key (e.g. "episodic_success") — for runs that flatten at log time
        #   3. Dot-form ("rate.allocated_bits_per_transition") — W&B API alt form
        # NOTE: SummarySubDict is not a subclass of dict, so isinstance(…, dict) would
        # wrongly evaluate to False — use hasattr check instead.
        eval_ns_raw = summary.get("eval/")
        eval_ns = eval_ns_raw if (eval_ns_raw is not None and hasattr(eval_ns_raw, "get")) else {}
        run_ids.append(run.id)
        print(f"  Run {run.name} (id={run.id})")
        for metric in METRICS_OF_INTEREST:
            # Use explicit None checks — do NOT use `or` since 0.0 is falsy.
            val = eval_ns.get(metric)
            if val is None:
                val = summary.get(metric)
            if val is None:
                val = summary.get(metric.replace("/", "."))
            if val is not None:
                try:
                    mv[metric].append(float(val))
                    print(f"    {metric}: {float(val):.4f}")
                except (TypeError, ValueError):
                    pass
            else:
                print(f"    {metric}: MISSING")

    mv = _correct_allocated_bits(mv)
    metrics_blob = _aggregate(mv)

    return {
        "ids":        run_ids,
        "name_count": len(run_ids),
        "metrics":    metrics_blob,
    }


def load_canonical_dd(canonical_glob: str, key: str) -> dict:
    """Load DDCL-CE (d,d) condition from the canonical all-methods heldout JSON."""
    import glob
    files = sorted(glob.glob(canonical_glob))
    if not files:
        print(f"WARNING: No canonical JSON found matching {canonical_glob}")
        return {"ids": [], "name_count": 0, "metrics": {}}
    latest = files[-1]
    print(f"\nReading DDCL-CE (d,d) from {latest}")
    with open(latest) as f:
        data = json.load(f)
    if key not in data:
        print(f"WARNING: key '{key}' not found in {latest}; keys: {list(data.keys())}")
        return {"ids": [], "name_count": 0, "metrics": {}}
    blob = data[key]
    print(f"  n_seeds={blob.get('name_count', blob.get('ids', '?'))}")
    m = blob.get("metrics", {})
    s = m.get("episodic_success", {})
    p = m.get("episodic_precision_success_0p02", {})
    print(f"  success={s.get('mean', '?'):.3f}  prec@0.02={p.get('mean', '?'):.3f}")
    return blob


def main() -> int:
    api = wandb.Api(timeout=60)

    output: dict = {}

    # ── (d,d) from canonical JSON ──────────────────────────────────────────
    output["ddcl_ce_dd"] = load_canonical_dd(CANONICAL_HELDOUT_GLOB, CANONICAL_CE_KEY)

    # ── Stochastic conditions from W&B ─────────────────────────────────────
    for label, infix, eval_mode, tgt_mode in STOCH_CONDITIONS:
        cond_data = fetch_condition(api, infix)
        cond_data["eval_mode"]   = eval_mode
        cond_data["target_mode"] = tgt_mode
        output[label] = cond_data

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n── Stochasticity-ablation heldout summary ───────────────────────")
    for label, cond_data in output.items():
        m = cond_data.get("metrics", {})
        s = m.get("episodic_success", {})
        p = m.get("episodic_precision_success_0p02", {})
        eval_mode = cond_data.get("eval_mode", "det")
        tgt_mode  = cond_data.get("target_mode", "det")
        print(
            f"  {label} (eval={eval_mode}, tgt={tgt_mode}): "
            f"success={s.get('mean', float('nan')):.3f}±{s.get('std', float('nan')):.3f} "
            f"prec@0.02={p.get('mean', float('nan')):.3f}±{p.get('std', float('nan')):.3f} "
            f"n={s.get('n', 0)}"
        )

    # ── Save JSON ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = OUT_DIR / f"wandb_toy_stoch_ablation_5seed_heldout_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
