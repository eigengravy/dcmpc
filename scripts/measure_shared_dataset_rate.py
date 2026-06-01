#!/usr/bin/env python3
"""E7 — Shared-dataset realized rate + matched-capacity analysis (C7/T6).

Verifies C7 (T6): own-policy empirical entropy is confounded by state-visitation;
the fair cross-method rate axis is realized code entropy on a SHARED input
distribution D_ref (same for every method). Also extracts matched-capacity
Pareto comparison points from the existing quantizer_pareto sweep.

Protocol:
  1. Build D_ref = uniform 30×30 grid over the toy reachable state box
     (pos_x ∈ [-1.15, 1.15], pos_y ∈ [-1.0, 1.0]), fixed auxiliary features.
  2. For each method: load the frozen encoder (eval mode, ε=0), encode D_ref,
     compute Σ_groups Shannon entropy H → bits/transition = H_total.
  3. Extract matched-capacity triplets from pareto_sweep_aggregated.json:
       Budget ≈ 256 bits: VQ-16, FSQ-[4,4], DDCL-s1.0 (best λ)
       Budget ≈ 384 bits: VQ-64, FSQ-[8,8], DDCL-s3.0 (best λ)
     For each triplet: report on-policy performance + realized rate on D_ref.
  4. Save all results to e7_shared_dataset_rate_<date>.json.

Expected (C7/T6 prediction):
  - DDCL realized rate << nominal (rate penalty drives |z|→0 → fewer active bins)
  - VQ/FSQ realized rate ≈ nominal (no compression force; tanh saturation is the
    only mechanism, but FSQ levels are small so saturation is limited)
  - At matched capacity, DDCL outperforms VQ/FSQ in prec@0.02 (Table 1-Pareto)

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/measure_shared_dataset_rate.py

Optional flags:
    --pareto_root  (default auto-detected from output/toy_runs/quantizer_pareto_*)
    --n_x          (default 30)  grid points along x axis
    --n_y          (default 30)  grid points along y axis
    --device       (default cpu)
    --output       (optional) JSON output path override
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from omegaconf import OmegaConf
from tensordict import TensorDict

try:
    from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
except ImportError:
    from torchrl.data import BoundedTensorSpec as Bounded
    from torchrl.data import CompositeSpec as Composite
    from torchrl.data import UnboundedContinuousTensorSpec as Unbounded

import config  # noqa: F401 — registers Hydra structured agent configs
from dcmpc import DCMPC, DCMPCConfig
from envs import make_env

# ── Constants ─────────────────────────────────────────────────────────────────
# Toy state box: obs = [pos_x, pos_y, goal_x=1.0, goal_y=0.0,
#                       gate_width=0.08, crossed_gate=0, collided=0]
OBS_DIM = 7
POS_X_LO, POS_X_HI = -1.15, 1.15
POS_Y_LO, POS_Y_HI = -1.00, 1.00
FIXED_FEATURES = [1.0, 0.0, 0.08, 0.0, 0.0]  # goal_x, goal_y, gate_width, crossed, collided

PARETO_JSON = Path("private/Metrics/Toy/QuantizerPareto/pareto_sweep_aggregated.json")
OUT_DIR = Path("private/Metrics/Toy/Corrected Reevaluation")

# ── Canonical paper methods (all completed paper-used toy seeds) ─────────────
# Paths are relative to the dcmpc/ root.  These are intentionally explicit:
# the paper tables should not silently change when old experimental folders are
# present in output/.
CANONICAL_METHOD_FAMILIES = [
    {
        "key": "ddcl_cosine",
        "label": "DDCL-Cos",
        "run_dirs": [
            "output/toy_runs/ddcl_objective_sweep_20260514_073320/hydra/"
            f"toy-precision-gate-final--ddcl_mse--jepa-cosine-det-eval-wavg-targets-s35-l1e3--s{s}"
            for s in [0, 1, 2]
        ] + [
            "output/toy_runs/ddcl_objective_completion_20260516_131900/hydra/"
            f"toy-precision-gate-final--ddcl_mse--jepa-cosine-det-eval-wavg-targets-s35-l1e3--s{s}"
            for s in [3, 4]
        ],
    },
    {
        "key": "ddcl_mse",
        "label": "DDCL-MSE",
        "run_dirs": [
            "output/toy_runs/ddcl_objective_sweep_20260514_073320/hydra/"
            f"toy-precision-gate-final--ddcl_mse--mse-det-eval-wavg-targets-s35-l1e3--s{s}"
            for s in [0, 1, 2]
        ] + [
            "output/toy_runs/ddcl_objective_completion_20260516_131900/hydra/"
            f"toy-precision-gate-final--ddcl_mse--mse-det-eval-wavg-targets-s35-l1e3--s{s}"
            for s in [3, 4]
        ],
    },
    {
        "key": "ddcl_soft_ce",
        "label": "DDCL-SCE",
        "run_dirs": [
            "output/toy_runs/ddcl_soft_ce_20260518_165818/hydra/"
            f"toy-precision-gate-final--ddcl_soft_ce--s{s}"
            for s in [0, 1, 2, 3, 4]
        ],
    },
    {
        "key": "ddcl_ce",
        "label": "DDCL-CE",
        "run_dirs": [
            "output/toy_runs/final_v1_20260512_234000/hydra/"
            f"toy-precision-gate-final--ddcl_ce--s{s}"
            for s in [0, 1, 2, 3, 4]
        ],
    },
    {
        "key": "fsq_ce",
        "label": "FSQ-CE",
        "run_dirs": [
            "output/toy_runs/final_v1_20260512_234000/hydra/"
            f"toy-precision-gate-final--dcmpc--s{s}"
            for s in [0, 1, 2, 3, 4]
        ],
    },
    {
        "key": "vq_ce",
        "label": "VQ-CE",
        "run_dirs": [
            "output/toy_runs/final_v1_20260512_234000/hydra/"
            f"toy-precision-gate-final--vq_ce--s{s}"
            for s in [0, 1, 2, 3, 4]
        ],
    },
    {
        "key": "continuous_mse",
        "label": "Cont-MSE",
        "run_dirs": [
            "output/toy_runs/final_v1_20260512_234000/hydra/"
            f"toy-precision-gate-final--continuous_mse--s{s}"
            for s in [0, 1, 2, 3, 4]
        ],
    },
]

STOCHASTICITY_METHOD_FAMILIES = [
    {
        "key": "ddcl_ce_stoch",
        "label": "DDCL-CE(s,s)",
        "run_dirs": [
            "output/toy_runs/ddcl_stoch_ablation_20260519_210503/hydra/"
            f"toy-precision-gate-final--ddcl_ce_stoch--s{s}"
            for s in [0, 1, 2, 3, 4]
        ],
    },
    {
        "key": "ddcl_ce_stoch_eval",
        "label": "DDCL-CE(s,d)",
        "run_dirs": [
            "output/toy_runs/ddcl_stoch_ablation_20260519_210503/hydra/"
            f"toy-precision-gate-final--ddcl_ce_stoch_eval--s{s}"
            for s in [0, 1, 2, 3, 4]
        ],
    },
    {
        "key": "ddcl_ce_stoch_tgt",
        "label": "DDCL-CE(d,s)",
        "run_dirs": [
            "output/toy_runs/ddcl_stoch_ablation_20260519_210503/hydra/"
            f"toy-precision-gate-final--ddcl_ce_stoch_tgt--s{s}"
            for s in [0, 1, 2, 3, 4]
        ],
    },
    {
        "key": "ddcl_cos_stoch_eval",
        "label": "DDCL-Cos(s,d)",
        "run_dirs": [
            "output/toy_runs/ddcl_stoch_ablation_20260519_210503/hydra/"
            f"toy-precision-gate-final--ddcl_cos_stoch_eval--s{s}"
            for s in [0, 1, 2, 3, 4]
        ],
    },
]

# ── Pareto methods to measure rate for (seed=0; covers both budget levels) ───
#   keyed by (method, condition) for JSON
PARETO_METHODS: list[dict] = []

# Budget 256: VQ-16, FSQ-[4,4], DDCL s1.0 (all four λ for Pareto curve)
PARETO_METHODS += [
    {
        "key": f"pareto_ddcl_s1p0_l{lam}",
        "label": f"DDCL s1.0 λ={lam}",
        "run_dir": f"output/toy_runs/quantizer_pareto_20260516_175600/hydra/"
                   f"toy-precision-gate-final--ddcl_cosine--ddcl-s1p0-d1p0-lambda{lam_tag}--s0",
        "pareto_key": f"s1_d1_l{lam}",
        "budget_bits": 256.0,
    }
    for lam, lam_tag in [("0.001", "0p001"), ("0.0003", "0p0003"),
                          ("0.003", "0p003"), ("0.0001", "0p0001")]
]
PARETO_METHODS += [
    {
        "key": "pareto_fsq_4x4",
        "label": "FSQ [4,4]",
        "run_dir": "output/toy_runs/quantizer_pareto_20260516_175600/hydra/"
                   "toy-precision-gate-final--dcmpc--fsq-4x4--s0",
        "pareto_key": "fsq_[4, 4]",
        "budget_bits": 256.0,
    },
    {
        "key": "pareto_vq_16",
        "label": "VQ-16",
        "run_dir": "output/toy_runs/quantizer_pareto_20260516_175600/hydra/"
                   "toy-precision-gate-final--vq_ce--vq-16--s0",
        "pareto_key": "vq_16",
        "budget_bits": 256.0,
    },
]

# Budget 384: VQ-64, FSQ-[8,8], DDCL s3.0 (all four λ)
PARETO_METHODS += [
    {
        "key": f"pareto_ddcl_s3p0_l{lam}",
        "label": f"DDCL s3.0 λ={lam}",
        "run_dir": f"output/toy_runs/quantizer_pareto_20260516_175600/hydra/"
                   f"toy-precision-gate-final--ddcl_cosine--ddcl-s3p0-d1p0-lambda{lam_tag}--s0",
        "pareto_key": f"s3_d1_l{lam}",
        "budget_bits": 384.0,
    }
    for lam, lam_tag in [("0.001", "0p001"), ("0.0003", "0p0003"),
                          ("0.003", "0p003"), ("0.0001", "0p0001")]
]
PARETO_METHODS += [
    {
        "key": "pareto_fsq_8x8",
        "label": "FSQ [8,8]",
        "run_dir": "output/toy_runs/quantizer_pareto_20260516_175600/hydra/"
                   "toy-precision-gate-final--dcmpc--fsq-8x8--s0",
        "pareto_key": "fsq_[8, 8]",
        "budget_bits": 384.0,
    },
    {
        "key": "pareto_vq_64",
        "label": "VQ-64",
        "run_dir": "output/toy_runs/quantizer_pareto_20260516_175600/hydra/"
                   "toy-precision-gate-final--vq_ce--vq-64--s0",
        "pareto_key": "vq_64",
        "budget_bits": 384.0,
    },
]


# ── Core helpers ──────────────────────────────────────────────────────────────

def build_dref(n_x: int = 30, n_y: int = 30, device: str = "cpu") -> TensorDict:
    """Uniform grid over toy reachable 2D state box with fixed auxiliary features.

    Returns a TensorDict with key 'state' and shape [N, OBS_DIM] where N = n_x * n_y.
    """
    x_vals = np.linspace(POS_X_LO, POS_X_HI, n_x)
    y_vals = np.linspace(POS_Y_LO, POS_Y_HI, n_y)
    xx, yy = np.meshgrid(x_vals, y_vals)      # each [n_y, n_x]
    pos_x = xx.ravel()                         # [N]
    pos_y = yy.ravel()
    N = len(pos_x)

    obs = np.zeros((N, OBS_DIM), dtype=np.float32)
    obs[:, 0] = pos_x
    obs[:, 1] = pos_y
    for i, val in enumerate(FIXED_FEATURES):
        obs[:, 2 + i] = val

    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    td = TensorDict({"state": obs_tensor}, batch_size=[N], device=device)
    return td


def load_agent_from_run_dir(run_dir: str, device: str) -> DCMPC | None:
    """Load a DCMPC agent from a run directory (contains checkpoint.pt + .hydra/).

    Returns None if the checkpoint cannot be loaded.
    """
    run_path = Path(run_dir)
    checkpoint_path = run_path / "checkpoint.pt"
    hydra_config = run_path / ".hydra" / "config.yaml"

    if not checkpoint_path.exists():
        print(f"    [SKIP] checkpoint not found: {checkpoint_path}")
        return None
    if not hydra_config.exists():
        print(f"    [SKIP] .hydra/config.yaml not found: {hydra_config}")
        return None

    try:
        train_cfg = OmegaConf.load(hydra_config)
        train_cfg.agent = OmegaConf.merge(
            OmegaConf.structured(DCMPCConfig), train_cfg.agent
        )
        train_cfg.device = device
        train_cfg.agent.device = device

        env = make_env(train_cfg, num_envs=1)
        obs_spec = env.observation_spec["observation"]
        if hasattr(obs_spec, "batch_size") and len(obs_spec.batch_size) > 0:
            obs_spec = obs_spec[0]
        act_spec = env.action_spec
        if hasattr(act_spec, "batch_size") and len(act_spec.batch_size) > 0:
            act_spec = act_spec[0]

        agent = DCMPC(train_cfg.agent, obs_spec=obs_spec, act_spec=act_spec).to(device)
        state_dict = torch.load(
            str(checkpoint_path), weights_only=True, map_location=torch.device(device)
        )
        agent.load_state_dict(state_dict["model"])
        agent.eval()
        return agent
    except Exception as e:
        print(f"    [ERROR] Failed to load {run_dir}: {e}")
        return None


@torch.no_grad()
def compute_realized_rate(
    agent: DCMPC,
    d_ref: TensorDict,
) -> dict:
    """Encode D_ref with ε=0 and compute Shannon code entropy.

    Returns:
        bits_per_transition: total H across all groups (bits)
        bits_per_group:      H averaged per group
        codebook_size:       per-group codebook size (None if continuous)
        n_groups:            number of quantizer groups (None if continuous)
        nominal_bits:        nominal capacity = n_groups * log2(codebook_size)
        compression_ratio:   nominal / realized (>1 means DDCL self-compressed)
        has_discrete:        False for continuous baseline (rate is NaN)
    """
    # Check if this is a discrete model
    if not agent.model._uses_discrete:
        return {
            "bits_per_transition": float("nan"),
            "bits_per_group": float("nan"),
            "codebook_size": None,
            "n_groups": None,
            "nominal_bits": float("nan"),
            "compression_ratio": float("nan"),
            "has_discrete": False,
        }

    # Forward pass through encoder with ε=0
    z_dict = agent.model.encode(d_ref, stochastic=False)

    indices = z_dict["indices"]             # [N, n_groups]
    if indices.ndim == 1:
        indices = indices.unsqueeze(-1)     # [N, 1]

    n_groups = indices.shape[-1]
    codebook_size = agent.model._quantizer.codebook_size

    # Σ_groups Shannon entropy (bits) — same formula as _empirical_entropy_bits
    total_H = 0.0
    for g in range(n_groups):
        group_tokens = indices[:, g].reshape(-1).to(torch.long)
        counts = torch.bincount(group_tokens, minlength=codebook_size)
        probs = counts.float() / counts.sum().clamp_min(1)
        probs = probs[probs > 0]
        total_H += float(-(probs * probs.log2()).sum().item())

    nominal_bits = n_groups * math.log2(codebook_size)
    compression_ratio = nominal_bits / max(total_H, 1e-9)

    return {
        "bits_per_transition": float(total_H),
        "bits_per_group": float(total_H) / n_groups,
        "codebook_size": int(codebook_size),
        "n_groups": int(n_groups),
        "nominal_bits": float(nominal_bits),
        "compression_ratio": float(compression_ratio),
        "has_discrete": True,
    }


def load_pareto_performance(pareto_json: Path) -> dict:
    """Load the pareto sweep aggregated JSON."""
    with open(pareto_json) as f:
        return json.load(f)


def extract_performance(pareto_data: dict, method_type: str, key: str) -> dict:
    """Extract performance metrics for a single Pareto condition."""
    # method_type is 'ddcl', 'vq', or 'fsq'
    # key is e.g. 's1_d1_l0.001', 'vq_16', 'fsq_[4, 4]'
    entry = pareto_data.get(method_type, {}).get(key, {})
    if not entry:
        return {}
    metrics = {}
    for m in ["episodic_success", "episodic_precision_success_0p02",
              "episodic_precision_success_0p04", "episodic_return",
              "rate/empirical_entropy_bits_per_transition",
              "rate/allocated_bits_per_transition",
              "codebook/usage_percent"]:
        if m in entry:
            metrics[m] = entry[m]
    return metrics


def _mean_ci95(values: list[float]) -> dict:
    """Return n/mean/std/95% t-CI for finite values."""
    vals = [float(v) for v in values if np.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "values": [], "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    if n == 1:
        return {"n": 1, "values": vals, "mean": vals[0], "std": 0.0, "ci95": 0.0}
    arr = np.array(vals, dtype=float)
    # t critical values for the small n used here. Fall back to normal approx.
    tcrit = {
        1: 12.706204736432095,
        2: 4.302652729749464,
        3: 3.182446305284263,
        4: 2.7764451051977987,
        5: 2.570581835636305,
        6: 2.4469118511449692,
        7: 2.3646242510102993,
        8: 2.3060041350333704,
        9: 2.2621571628540993,
        10: 2.2281388519649385,
    }.get(n - 1, 1.959963984540054)
    std = float(arr.std(ddof=1))
    ci95 = float(tcrit * std / math.sqrt(n))
    return {"n": n, "values": vals, "mean": float(arr.mean()), "std": std, "ci95": ci95}


def _summarise_seed_rates(seed_entries: list[dict]) -> dict:
    keys = [
        "bits_per_transition",
        "bits_per_group",
        "nominal_bits",
        "compression_ratio",
    ]
    summary = {}
    for key in keys:
        summary[key] = _mean_ci95([
            entry["rate"].get(key, float("nan"))
            for entry in seed_entries
            if entry.get("rate", {}).get("has_discrete", False)
        ])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_x", type=int, default=30, help="Grid points along x")
    parser.add_argument("--n_y", type=int, default=30, help="Grid points along y")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="Override output JSON path")
    parser.add_argument("--pareto_only", action="store_true",
                        help="Only run Pareto methods, skip canonical")
    parser.add_argument("--canonical_only", action="store_true",
                        help="Only run canonical methods, skip Pareto")
    args = parser.parse_args()

    os.chdir(_ROOT)

    print(f"Building D_ref grid ({args.n_x}×{args.n_y} = {args.n_x * args.n_y} points) ...")
    d_ref = build_dref(args.n_x, args.n_y, device=args.device)
    N = args.n_x * args.n_y
    print(f"  D_ref shape: {d_ref['state'].shape}  (pos_x ∈ [{POS_X_LO}, {POS_X_HI}], "
          f"pos_y ∈ [{POS_Y_LO}, {POS_Y_HI}])")

    pareto_data = load_pareto_performance(PARETO_JSON)

    results = {
        "meta": {
            "n_x": args.n_x,
            "n_y": args.n_y,
            "n_obs": N,
            "obs_dim": OBS_DIM,
            "rate_metric": "R_m(D_ref) = sum_g H[codes_g(s)], s ~ shared toy grid, epsilon=0",
            "canonical_policy": "aggregate all listed completed paper-used toy seeds; continuous models have no discrete rate",
            "fixed_features": {"goal_x": 1.0, "goal_y": 0.0,
                                "gate_width": 0.08, "crossed_gate": 0, "collided": 0},
            "state_box": {"pos_x": [POS_X_LO, POS_X_HI],
                          "pos_y": [POS_Y_LO, POS_Y_HI]},
        },
        "canonical": {},
        "pareto": {},
    }

    # ── Canonical methods ──────────────────────────────────────────────────────
    if not args.pareto_only:
        print("\n── Canonical methods (realized rate on D_ref; all listed seeds) ───")
        for spec in CANONICAL_METHOD_FAMILIES:
            seed_entries = []
            print(f"\n  [{spec['label']}] {len(spec['run_dirs'])} seed/checkpoint dirs")
            for run_dir in spec["run_dirs"]:
                print(f"    loading {run_dir} ...")
                agent = load_agent_from_run_dir(run_dir, args.device)
                if agent is None:
                    seed_entries.append({
                        "run_dir": run_dir,
                        "status": "missing_or_load_failed",
                        "rate": {"has_discrete": False, "bits_per_transition": float("nan")},
                    })
                    continue
                rate_info = compute_realized_rate(agent, d_ref)
                if rate_info["has_discrete"]:
                    print(f"      H = {rate_info['bits_per_transition']:.2f} bits/trans")
                else:
                    print("      continuous (no quantizer)")
                seed_entries.append({
                    "run_dir": run_dir,
                    "status": "ok",
                    "rate": rate_info,
                })
                del agent

            ok_entries = [e for e in seed_entries if e.get("status") == "ok"]
            summary = _summarise_seed_rates(ok_entries)
            if ok_entries and ok_entries[0]["rate"].get("has_discrete", False):
                b = summary["bits_per_transition"]
                print(f"    mean H = {b['mean']:.2f} ± {b['ci95']:.2f} bits/trans (n={b['n']})")
                representative_rate = dict(ok_entries[0]["rate"])
                representative_rate["bits_per_transition"] = b["mean"]
                representative_rate["bits_per_transition_ci95"] = b["ci95"]
            else:
                representative_rate = {
                    "bits_per_transition": float("nan"),
                    "bits_per_transition_ci95": float("nan"),
                    "has_discrete": False,
                }
            results["canonical"][spec["key"]] = {
                "label": spec["label"],
                "run_dirs": spec["run_dirs"],
                "seeds": seed_entries,
                "summary": summary,
                # Backwards-compatible convenience field for older consumers.
                "rate": representative_rate,
            }

        print("\n── Stochasticity ablation methods (realized rate on D_ref) ────────")
        results["stochasticity"] = {}
        for spec in STOCHASTICITY_METHOD_FAMILIES:
            seed_entries = []
            print(f"\n  [{spec['label']}] {len(spec['run_dirs'])} seed/checkpoint dirs")
            for run_dir in spec["run_dirs"]:
                agent = load_agent_from_run_dir(run_dir, args.device)
                if agent is None:
                    seed_entries.append({
                        "run_dir": run_dir,
                        "status": "missing_or_load_failed",
                        "rate": {"has_discrete": False, "bits_per_transition": float("nan")},
                    })
                    continue
                rate_info = compute_realized_rate(agent, d_ref)
                print(f"      H = {rate_info.get('bits_per_transition', float('nan')):.2f} bits/trans")
                seed_entries.append({"run_dir": run_dir, "status": "ok", "rate": rate_info})
                del agent
            summary = _summarise_seed_rates([e for e in seed_entries if e.get("status") == "ok"])
            b = summary["bits_per_transition"]
            print(f"    mean H = {b['mean']:.2f} ± {b['ci95']:.2f} bits/trans (n={b['n']})")
            results["stochasticity"][spec["key"]] = {
                "label": spec["label"],
                "run_dirs": spec["run_dirs"],
                "seeds": seed_entries,
                "summary": summary,
            }

    # ── Pareto methods ─────────────────────────────────────────────────────────
    if not args.canonical_only:
        print("\n── Pareto methods (realized rate + performance) ──────────────────")
        for spec in PARETO_METHODS:
            print(f"\n  [{spec['label']}] loading {spec['run_dir']} ...")
            agent = load_agent_from_run_dir(spec["run_dir"], args.device)
            if agent is None:
                # Still include performance data even if rate unavailable
                rate_info = {"has_discrete": False, "bits_per_transition": float("nan"),
                             "nominal_bits": float("nan"), "compression_ratio": float("nan")}
            else:
                rate_info = compute_realized_rate(agent, d_ref)
                del agent

            if rate_info["has_discrete"]:
                print(f"    realized H = {rate_info['bits_per_transition']:.2f} bits/trans"
                      f"  (nominal {rate_info['nominal_bits']:.1f},"
                      f" compression {rate_info['compression_ratio']:.2f}×)")

            # Attach Pareto performance
            pareto_key = spec.get("pareto_key", "")
            if "ddcl" in spec["key"]:
                perf = extract_performance(pareto_data, "ddcl", pareto_key)
            elif "vq" in spec["key"]:
                perf = extract_performance(pareto_data, "vq", pareto_key)
            elif "fsq" in spec["key"]:
                perf = extract_performance(pareto_data, "fsq", pareto_key)
            else:
                perf = {}

            if perf:
                s = perf.get("episodic_success", {}).get("mean", float("nan"))
                p = perf.get("episodic_precision_success_0p02", {}).get("mean", float("nan"))
                print(f"    success={s:.3f}  prec@0.02={p:.3f}"
                      f"  budget={spec.get('budget_bits', '?')} bits")

            results["pareto"][spec["key"]] = {
                "label": spec["label"],
                "run_dir": spec["run_dir"],
                "budget_bits": spec.get("budget_bits"),
                "pareto_key": pareto_key,
                "rate": rate_info,
                "performance": perf,
            }

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n── Summary: Realized vs Nominal rate on D_ref ────────────────────")
    print(f"  {'Method':<30} {'Realized H':>12} {'Nominal':>10} {'Ratio':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*8}")
    for key, entry in results["canonical"].items():
        r = entry.get("rate", {})
        if r["has_discrete"]:
            ci = r.get("bits_per_transition_ci95", 0.0)
            print(f"  {entry['label']:<30} {r['bits_per_transition']:>12.1f} "
                  f"{r['nominal_bits']:>10.1f} {r['compression_ratio']:>8.2f}×"
                  f"  ±{ci:.1f}")
        else:
            print(f"  {entry['label']:<30} {'(continuous)':>12}")

    print()
    print("── Matched-capacity comparison ───────────────────────────────────")
    for budget in [256.0, 384.0]:
        print(f"\n  Budget ≈ {budget:.0f} bits:")
        print(f"  {'Method':<28} {'Succ':>6} {'Prec@0.02':>10} {'Realized H':>12}")
        for key, entry in results["pareto"].items():
            if abs((entry.get("budget_bits") or 0) - budget) > 5:
                continue
            r = entry["rate"]
            p = entry.get("performance", {})
            s = p.get("episodic_success", {}).get("mean", float("nan"))
            prec = p.get("episodic_precision_success_0p02", {}).get("mean", float("nan"))
            bits = r.get("bits_per_transition", float("nan"))
            print(f"  {entry['label']:<28} {s:>6.3f} {prec:>10.3f} {bits:>12.1f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = Path(args.output) if args.output else OUT_DIR / f"e7_shared_dataset_rate_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
