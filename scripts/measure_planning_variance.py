#!/usr/bin/env python3
"""E6 — Direct rollout-variance probe for T1 (C1) verification.

Empirically verifies T1: stochastic DDCL quantization inside MPPI imagined
rollouts injects additive per-trajectory return variance (Var_quant > 0) that
is independent of action quality and unaffected by more MPPI samples. Setting
ε=0 (deterministic eval) eliminates this variance entirely.

Protocol:
  1. Load a trained DDCL-Cos checkpoint (no training, eval mode).
  2. Get an initial latent state z by encoding one env reset observation.
  3. Fix K action sequences (random, held constant across all repetitions).
  4. For each of the two conditions:
       - STOCHASTIC: call _single_estimate_value K_rep times per action sequence
         with stochastic=True → draw fresh ε at each trans() step.
       - DETERMINISTIC: call K_rep times with stochastic=False (ε=0) →
         return should be identical across repetitions.
  5. Compute per-sequence variance, then mean over sequences.
  6. Report Var_quant (stochastic), Var_det (≈ 0), and their ratio.

Expected (T1 prediction):
  - Var_det ≈ 0 (up to floating-point noise)
  - Var_quant > 0 (scales with Σ_t γ^2t · L²c²σ²_ε)
  - Ratio Var_quant / (Var_det + ε) >> 1

Usage (from dcmpc/ directory):
    conda run -n ddcl_mbrl python scripts/measure_planning_variance.py \\
        --checkpoint output/toy_runs/ddcl_objective_sweep_20260514_073320/hydra/\\
toy-precision-gate-final--ddcl_mse--jepa-cosine-det-eval-wavg-targets-s35-l1e3--s0/checkpoint.pt

Optional flags:
    --k_sequences  (default 20)   number of distinct action sequences
    --k_rep        (default 200)  repetitions per sequence for variance estimation
    --seed         (default 0)    RNG seed
    --device       (default cpu)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from omegaconf import OmegaConf
try:
    from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
except ImportError:
    from torchrl.data import BoundedTensorSpec as Bounded  # older torchrl alias
    from torchrl.data import CompositeSpec as Composite
    from torchrl.data import UnboundedContinuousTensorSpec as Unbounded

import config  # noqa: F401 — registers Hydra structured agent configs
from dcmpc import DCMPC, DCMPCConfig
from envs import make_env


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "output/toy_runs/ddcl_objective_sweep_20260514_073320/hydra/"
    "toy-precision-gate-final--ddcl_mse--jepa-cosine-det-eval-wavg-targets-s35-l1e3--s0/"
    "checkpoint.pt"
)


def load_agent(checkpoint_path: str, device: str) -> tuple:
    """Load DCMPC agent + env from a checkpoint path (relative to dcmpc/)."""
    checkpoint_path = str(Path(_ROOT) / checkpoint_path)
    cfg_dir = str(Path(checkpoint_path).parent) + "/"
    train_cfg = OmegaConf.load(Path(cfg_dir) / ".hydra" / "config.yaml")
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
        checkpoint_path, weights_only=True, map_location=torch.device(device)
    )
    agent.load_state_dict(state_dict["model"])
    agent.eval()
    print(f"Loaded: {checkpoint_path}")
    print(f"  quantizer: {agent.model._quantizer}")
    print(f"  ddcl_deterministic_eval: {agent.cfg.ddcl_deterministic_eval}")
    return agent, env


@torch.no_grad()
def probe_variance(
    agent: DCMPC,
    env,
    k_sequences: int = 20,
    k_rep: int = 200,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """
    Returns a dict with scalar statistics on stochastic vs. deterministic
    imagined return variance.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Get one starting latent state z ──────────────────────────────────────
    td = env.reset()
    # td["observation"] is a TensorDict of obs-type keys (e.g. {"state": ...})
    obs = td["observation"].to(device)
    with torch.no_grad():
        # encode() lives on WorldModel (agent.model), not the outer DCMPC wrapper
        z_dict = agent.model.encode(obs, stochastic=False).to(torch.float)
    # z_dict is a TensorDict; keep as-is (DCMPC.encode returns a TensorDict)
    print(f"\nStarting latent z shape: {z_dict['codes'].shape}")

    H = agent.cfg.plan_horizon
    act_dim = agent.act_dim
    print(f"  plan_horizon={H}, act_dim={act_dim}, k_sequences={k_sequences}, k_rep={k_rep}")

    # ── Sample k_sequences distinct action sequences ──────────────────────────
    # _single_estimate_value expects actions[t] to have shape [batch, act_dim]
    # (same batch dim as z["codes"]). Keep explicit batch=1 dim.
    # Shape: [H, k_sequences, 1, act_dim]
    action_seqs = torch.rand(H, k_sequences, 1, act_dim, device=device) * 2 - 1
    action_seqs = action_seqs.clamp(-1, 1)

    stoch_var_per_seq  = []
    det_var_per_seq    = []
    stoch_mean_per_seq = []
    det_mean_per_seq   = []

    # Pre-generate the terminal action for each sequence (deterministic pi output
    # with eval_mode=True to avoid exploration noise in terminal Q-estimate).
    # This isolates trans() dither as the ONLY source of variance.
    z_exp0 = z_dict.expand(1)
    terminal_actions = []
    for seq_idx in range(k_sequences):
        actions_h_init = action_seqs[:, seq_idx, :, :]  # [H, 1, act_dim]
        # Simulate the deterministic rollout to get a terminal latent
        z_t = z_exp0
        for t in range(H):
            z_t = agent.model.trans(
                z_t["codes"], actions_h_init[t],
                unc_prop_mode=agent.cfg.plan_unc_prop_mode,
                stochastic=False,
            )
        # Deterministic terminal action (no exploration noise)
        a_term = agent.pi(z_t["codes"], eval_mode=True)  # [1, act_dim]
        terminal_actions.append(a_term.detach().clone())

    def estimate_return(z_start, actions_h, a_terminal, stochastic: bool) -> float:
        """
        Compute H-step discounted return + terminal Q-value.
        All actions including terminal are FIXED; only trans() stochasticity varies.
        """
        G, discount = 0.0, 1.0
        z = z_start.clone()
        for t in range(H):
            r = agent.model.reward(z["codes"], actions_h[t])
            z = agent.model.trans(
                z["codes"], actions_h[t],
                unc_prop_mode=agent.cfg.plan_unc_prop_mode,
                stochastic=stochastic,
            )
            G = G + discount * r
            discount = discount * agent.cfg.rho
        q_val = agent.Q(z["codes"], a_terminal, return_type="avg")
        return (G + discount * q_val).item()

    for seq_idx in range(k_sequences):
        actions_h = action_seqs[:, seq_idx, :, :]  # [H, 1, act_dim]
        a_term = terminal_actions[seq_idx]  # [1, act_dim]

        stoch_returns = []
        det_returns   = []

        for _ in range(k_rep):
            z_exp = z_dict.expand(1)

            # Stochastic: fresh ε at each trans() step
            val_s = estimate_return(z_exp, actions_h, a_term, stochastic=True)
            stoch_returns.append(val_s)

            # Deterministic: ε=0 at every trans() step → EXACT return, Var = 0
            val_d = estimate_return(z_exp, actions_h, a_term, stochastic=False)
            det_returns.append(val_d)

        stoch_arr = np.array(stoch_returns)
        det_arr   = np.array(det_returns)

        stoch_var_per_seq.append(np.var(stoch_arr, ddof=1))
        det_var_per_seq.append(np.var(det_arr, ddof=1))
        stoch_mean_per_seq.append(np.mean(stoch_arr))
        det_mean_per_seq.append(np.mean(det_arr))

    stoch_var = np.mean(stoch_var_per_seq)
    det_var   = np.mean(det_var_per_seq)
    ratio     = stoch_var / (det_var + 1e-12)

    # Also measure std of imagined return across sequences (signal variability)
    signal_std = np.std(stoch_mean_per_seq, ddof=1)

    results = {
        "var_stochastic":      float(stoch_var),
        "var_deterministic":   float(det_var),
        "std_stochastic":      float(np.sqrt(stoch_var)),
        "std_deterministic":   float(np.sqrt(det_var)),
        "var_ratio":           float(ratio),
        "signal_std_across_seqs": float(signal_std),
        "k_sequences":         k_sequences,
        "k_rep":               k_rep,
        "plan_horizon":        H,
        "seed":                seed,
        # Per-sequence breakdown
        "per_seq_stoch_var":   [float(v) for v in stoch_var_per_seq],
        "per_seq_det_var":     [float(v) for v in det_var_per_seq],
        "per_seq_stoch_mean":  [float(v) for v in stoch_mean_per_seq],
        "per_seq_det_mean":    [float(v) for v in det_mean_per_seq],
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Relative (from dcmpc/) path to checkpoint.pt",
    )
    parser.add_argument("--k_sequences", type=int, default=20)
    parser.add_argument("--k_rep", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    os.chdir(_ROOT)  # ensure relative paths work

    agent, env = load_agent(args.checkpoint, args.device)

    print(f"\nRunning variance probe ({args.k_sequences} seqs × {args.k_rep} reps) ...")
    results = probe_variance(
        agent, env,
        k_sequences=args.k_sequences,
        k_rep=args.k_rep,
        seed=args.seed,
        device=args.device,
    )

    # ── Print headline ────────────────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────")
    print("E6 Rollout-variance probe results (T1 / C1 verification)")
    print("──────────────────────────────────────────────────────────────")
    print(f"  Var_stochastic (ε~U)  = {results['var_stochastic']:.4f}")
    print(f"  Var_deterministic (ε=0) = {results['var_deterministic']:.6f}")
    print(f"  Std_stochastic         = {results['std_stochastic']:.4f}")
    print(f"  Std_deterministic      = {results['std_deterministic']:.6f}")
    print(f"  Var_ratio              = {results['var_ratio']:.1f}×")
    print(f"  Signal std (across seqs) = {results['signal_std_across_seqs']:.4f}")
    print()
    print("  T1 prediction: Var_quant > 0 (stoch >> det), det ≈ 0.")
    verified = results["var_stochastic"] > 10 * results["var_deterministic"]
    print(f"  Verified (stoch > 10× det): {'YES ✓' if verified else 'NO ✗'}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")

    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
