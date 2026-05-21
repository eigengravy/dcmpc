#!/usr/bin/env python3
"""Correct rate/allocated_bits_per_transition in paper metric JSONs.

All DDCL runs before 2026-05-21 logged rate/allocated_bits_per_transition
using the signed formula log₂(2|z|/δ+1), which inflates by ~36% vs the
DDCL-paper unsigned formula log₂(|z|/δ+1) used in training comm_loss.

This script reads the locally cached wandb-summary.json files for every
DDCL heldout-eval run and replaces rate/allocated_bits_per_transition with:
    corrected = rate/ddcl_loss_bits_per_dim × n_total_dims
where n_total_dims = ddcl_signed_bits_per_transition / ddcl_signed_bits_per_dim
(derived per-run so it works across architectures).

Also corrects the pareto aggregated JSON using the same approach.

Non-DDCL methods are untouched (their allocated_bits = max_bits, formula-free).

Usage (from dcmpc/ directory):
    python scripts/correct_allocated_bits_in_jsons.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_wandb_summary(path: Path) -> dict:
    """Read a wandb-summary.json and return the eval/ sub-dict."""
    with open(path) as f:
        summary = json.load(f)
    # Heldout eval runs log under "eval/"
    ev = summary.get("eval/", {})
    if not ev:
        # Training runs log under "eval_best/"
        ev = summary.get("eval_best/", {})
    return ev


def corrected_allocated_bits(ev: dict) -> float | None:
    """Compute corrected allocated bits/transition from unsigned per-dim metric."""
    loss_per_dim = ev.get("rate/ddcl_loss_bits_per_dim")
    signed_per_dim = ev.get("rate/ddcl_signed_bits_per_dim")
    signed_total = ev.get("rate/ddcl_signed_bits_per_transition")
    if loss_per_dim is None or signed_per_dim is None or signed_total is None:
        return None
    if abs(signed_per_dim) < 1e-9:
        return None
    n_total_dims = round(signed_total / signed_per_dim)
    return loss_per_dim * n_total_dims


def extract_seed(path: Path) -> int:
    """Extract seed number from a wandb-summary.json path."""
    # Paths end with something like --s0/wandb/run-.../files/wandb-summary.json
    # or seed=0 in the directory name
    for part in reversed(path.parts):
        m = re.search(r"--s(\d+)$", part)
        if m:
            return int(m.group(1))
        m = re.search(r"seed[=_]?(\d+)", part, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return -1


# ── Per-method run locators ───────────────────────────────────────────────────
# Maps method key (as used in the heldout JSON) to (glob-pattern, run-root)

ROOT = Path("output/toy_runs")

METHOD_SOURCES: dict[str, list[tuple[str, str]]] = {
    # method_key: list of (hydra_subdir_glob, run_root_dir)
    "ddcl_cosine": [
        ("*jepa-cosine-det-eval-wavg-targets-s35-l1e3--s*",
         "objective_checkpoint_heldout_eval_20260516_173200/hydra"),
    ],
    "ddcl_mse": [
        ("*mse-det-eval-wavg-targets-s35-l1e3--s*",
         "objective_checkpoint_heldout_eval_20260516_173200/hydra"),
    ],
    "ddcl_ce": [
        ("final-v1--toy-precision-gate-final--ddcl_ce--s*",
         "paper_checkpoint_heldout_eval_20260516_174100/hydra"),
    ],
    "ddcl_ce_repair": [
        ("best-ce--toy-precision-gate-final--ddcl_ce--ce-det-eval-wavg-targets-s35-l1e3--s*",
         "paper_checkpoint_heldout_eval_20260516_174100/hydra"),
    ],
    "ddcl_soft_ce": [
        ("soft_ce_heldout--toy-precision-gate-final--ddcl_soft_ce--s*",
         "soft_ce_heldout_eval_20260519_210319/hydra"),
    ],
}


def collect_run_summaries(method_key: str) -> dict[int, float]:
    """Return {seed: corrected_allocated_bits} for a DDCL method."""
    seed_to_bits: dict[int, float] = {}
    for glob_pat, subdir in METHOD_SOURCES[method_key]:
        for summary_path in sorted(
            ROOT.glob(f"{subdir}/{glob_pat}/wandb/run-*/files/wandb-summary.json")
        ):
            seed = extract_seed(summary_path.parent.parent.parent.parent)
            ev = read_wandb_summary(summary_path)
            bits = corrected_allocated_bits(ev)
            if bits is None:
                print(f"  WARNING: missing ddcl_loss_bits metrics in {summary_path}")
                continue
            # Keep the latest run if multiple wandb dirs exist for same seed
            if seed not in seed_to_bits or True:  # always take last sorted
                seed_to_bits[seed] = bits
            print(f"  {method_key} seed={seed}: corrected={bits:.2f} bits "
                  f"(from {summary_path.parent.parent.parent.parent.name})")
    return seed_to_bits


# ── Patch heldout JSON files ──────────────────────────────────────────────────

HELDOUT_JSONS = [
    Path("private/Metrics/Toy/Corrected Reevaluation/"
         "wandb_toy_all_methods_5seed_heldout_20260517.json"),
    Path("private/Metrics/Toy/Corrected Reevaluation/"
         "wandb_toy_soft_ce_5seed_heldout_20260519.json"),
]

DDCL_METHODS = {"ddcl_cosine", "ddcl_mse", "ddcl_ce", "ddcl_ce_repair", "ddcl_soft_ce"}

changes: dict[str, dict] = {}  # method -> {seed: old_val, new_val}


def patch_heldout_json(json_path: Path) -> bool:
    print(f"\n=== Patching {json_path.name} ===")
    with open(json_path) as f:
        data = json.load(f)

    modified = False
    for method, entry in data.items():
        if method not in DDCL_METHODS:
            continue
        metrics = entry.get("metrics", {})
        if "rate/allocated_bits_per_transition" not in metrics:
            print(f"  {method}: no allocated_bits metric, skipping")
            continue

        print(f"\n  Method: {method}")
        seed_to_bits = collect_run_summaries(method)
        if not seed_to_bits:
            print(f"  WARNING: no runs found for {method}, skipping")
            continue

        old_vals = metrics["rate/allocated_bits_per_transition"]["values"]
        new_vals = list(old_vals)  # copy

        # The values list is indexed by seed_index (0..N-1), which corresponds
        # to the order the extraction script found runs (sorted by run name).
        # We have seeds sorted by seed number — map seed_index to seed number.
        sorted_seeds = sorted(seed_to_bits.keys())
        for idx, seed in enumerate(sorted_seeds):
            if idx < len(new_vals):
                old = new_vals[idx]
                new = seed_to_bits[seed]
                new_vals[idx] = new
                print(f"    seed_idx={idx} (seed={seed}): {old:.2f} → {new:.2f}")

        import numpy as np
        old_mean = metrics["rate/allocated_bits_per_transition"]["mean"]
        old_std  = metrics["rate/allocated_bits_per_transition"].get("std", None)
        new_mean = float(np.mean(new_vals))
        new_std  = float(np.std(new_vals, ddof=1))

        metrics["rate/allocated_bits_per_transition"]["values"] = new_vals
        metrics["rate/allocated_bits_per_transition"]["mean"]   = new_mean
        metrics["rate/allocated_bits_per_transition"]["std"]    = new_std
        if "n" not in metrics["rate/allocated_bits_per_transition"]:
            metrics["rate/allocated_bits_per_transition"]["n"] = len(new_vals)
        print(f"    mean: {old_mean:.2f} → {new_mean:.2f}  (std: {old_std} → {new_std:.2f})")
        modified = True

    if modified:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved → {json_path}")
    else:
        print("  No changes made.")
    return modified


# ── Patch pareto JSON (DDCL entries from training eval_best/) ─────────────────

PARETO_JSON = Path("private/Metrics/Toy/QuantizerPareto/pareto_sweep_aggregated.json")

# Pareto DDCL runs are training runs that log under eval_best/
# We need to find their wandb-summary.json files in the training run directories

PARETO_DDCL_ROOT = ROOT / "quantizer_pareto_20260516_175600" / "hydra"


def patch_pareto_json() -> bool:
    if not PARETO_JSON.exists():
        print(f"\n  Pareto JSON not found: {PARETO_JSON}")
        return False
    if not PARETO_DDCL_ROOT.exists():
        print(f"\n  Pareto run root not found: {PARETO_DDCL_ROOT}")
        return False

    print(f"\n=== Patching {PARETO_JSON.name} ===")
    with open(PARETO_JSON) as f:
        data = json.load(f)

    if "ddcl" not in data:
        print("  No ddcl family in pareto JSON.")
        return False

    # Collect corrected values per config_key per seed
    # Pareto training run hydra dirs are named like:
    # toy-precision-gate-final--ddcl_cosine--<variant>--s<seed>
    # But the agent key in the pareto is config_key e.g. "s3.5_d1_l0.001"

    import re as _re

    def get_config_key_from_path(p: Path) -> str | None:
        """Try to extract scale/delta/lambda from the hydra dir name."""
        name = p.name
        # Pattern like: scale3.5-delta1.0-lambda0.001 or s3.5_d1_l0.001
        m_s = _re.search(r"s(?:cale)?[_=]?(\d+(?:\.\d+)?)", name)
        m_d = _re.search(r"d(?:elta)?[_=]?(\d+(?:\.\d+)?)", name)
        m_l = _re.search(r"l(?:ambda)?[_=]?([\d.e+-]+)", name)
        if m_s and m_l:
            scale = m_s.group(1).rstrip("0").rstrip(".")
            lam   = m_l.group(1)
            delta = m_d.group(1).rstrip("0").rstrip(".") if m_d else "1"
            return f"s{scale}_d{delta}_l{lam}"
        return None

    # Collect: {config_key: {seed: corrected_bits}}
    pareto_corrected: dict[str, dict[int, float]] = {}

    for summary_path in sorted(PARETO_DDCL_ROOT.glob(
        "*/wandb/run-*/files/wandb-summary.json"
    )):
        hydra_dir = summary_path.parent.parent.parent.parent
        ck = get_config_key_from_path(hydra_dir)
        if ck is None:
            continue
        seed = extract_seed(hydra_dir)
        ev = read_wandb_summary(summary_path)
        bits = corrected_allocated_bits(ev)
        if bits is None:
            continue
        if ck not in pareto_corrected:
            pareto_corrected[ck] = {}
        pareto_corrected[ck][seed] = bits

    if not pareto_corrected:
        print("  WARNING: could not extract any pareto DDCL runs from local files.")
        return False

    import numpy as np
    modified = False
    for ck, seed_bits in pareto_corrected.items():
        if ck not in data["ddcl"]:
            continue
        entry = data["ddcl"][ck]
        if "rate/allocated_bits_per_transition" not in entry:
            continue
        old_entry = entry["rate/allocated_bits_per_transition"]
        vals = list(seed_bits.values())
        new_mean = float(np.mean(vals))
        new_std  = float(np.std(vals, ddof=0))
        old_mean = old_entry["mean"]
        print(f"  {ck}: {old_mean:.2f} → {new_mean:.2f}  (n={len(vals)})")
        entry["rate/allocated_bits_per_transition"] = {
            "mean": new_mean, "std": new_std, "n": len(vals)
        }
        modified = True

    if modified:
        with open(PARETO_JSON, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n  Saved → {PARETO_JSON}")
    return modified


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np  # noqa: F401 (used inside functions)

    for jp in HELDOUT_JSONS:
        if jp.exists():
            patch_heldout_json(jp)
        else:
            print(f"\nSkipping (not found): {jp}")

    patch_pareto_json()
    print("\nDone. Re-run make_paper_plots.py to regenerate figures.")
