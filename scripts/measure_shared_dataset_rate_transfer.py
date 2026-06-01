#!/usr/bin/env python3
"""Shared-dataset realized rate R_m for transfer environments.

Mirrors the toy R_m computation (measure_shared_dataset_rate.py) but for
high-dimensional transfer tasks where a uniform grid is infeasible. Instead,
we pool trained-policy eval rollouts from ALL compared methods into one
shared probe set D*_task, then measure each encoder's code entropy on that
same set.

Why pooled trained-policy states (not random):
  The encoder converged on states from the training policy distribution.
  Random-policy states are out-of-distribution and would measure extrapolation
  behavior, not real encoder capacity. Pooling across all methods ensures no
  single method's visitation defines the axis — the same logic as the toy
  uniform grid, adapted to spaces where we can't grid but can pool.

Two-phase workflow:
  Phase 1 — build_probe: download checkpoints from W&B, roll out each agent,
            pool observations, subsample to fixed N, save D*_task to disk.
  Phase 2 — measure_rate: load D*_task, encode with each agent (eps=0),
            compute per-group Shannon entropy = R_m.

Run both phases:
    conda run -n ddcl_mbrl python scripts/measure_shared_dataset_rate_transfer.py \
        --env mw-button-press --wandb_project f20210966/dcmpc

Run phase 2 only (probe already built):
    conda run -n ddcl_mbrl python scripts/measure_shared_dataset_rate_transfer.py \
        --env mw-button-press --probe_only --wandb_project f20210966/dcmpc
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from omegaconf import OmegaConf
from tensordict import TensorDict

import config  # noqa: F401 — registers Hydra structured agent configs
from dcmpc import DCMPC, DCMPCConfig
from envs import make_env

# ── Method specifications for each transfer environment ───────────────────────
# Each entry: W&B run name pattern, non-archive, finished, paper-used seeds.
# The script downloads the best checkpoint from each run.

def _make_method_specs(env_name: str) -> list[dict]:
    """Generate the four standard method specs for a transfer environment."""
    base_filter = {
        "config.env_name": env_name,
        "state": "finished",
        "tags": {"$nin": ["archive"]},
    }
    return [
        {
            "key": "ddcl_cosine",
            "label": "DDCL-Cos",
            "wandb_filter": dict(base_filter),
            "match_tags": ["quantizer=ddcl", "consistency=cosine", "protocol=stable"],
            "match_config": {"agent.ddcl_lambda": 0.001},
        },
        {
            "key": "ddcl_ce_sd",
            "label": "DDCL-CE(s,d)",
            "wandb_filter": dict(base_filter),
            "match_tags": ["quantizer=ddcl", "consistency=cross-entropy", "protocol=default"],
            "match_config": {"agent.ddcl_lambda": 0.001},
        },
        {
            "key": "fsq_ce",
            "label": "FSQ-CE",
            "wandb_filter": dict(base_filter),
            "match_tags": ["quantizer=fsq", "consistency=cross-entropy"],
            "match_config": {},
        },
        {
            "key": "vq_ce",
            "label": "VQ-CE",
            "wandb_filter": dict(base_filter),
            "match_tags": ["quantizer=vq", "consistency=cross-entropy"],
            "match_config": {},
        },
    ]


# W&B env_name values: DMControl uses the task name without domain prefix
# (e.g. "walker" not "walker-walk"); MetaWorld uses "mw-button-press".
TRANSFER_METHODS = {
    "mw-button-press": _make_method_specs("mw-button-press"),
    "walker-walk":     _make_method_specs("walker"),
    "reacher-hard":    _make_method_specs("reacher"),
    "dog-run":         _make_method_specs("dog"),
    "humanoid-walk":   _make_method_specs("humanoid"),
}

# ── Constants ─────────────────────────────────────────────────────────────────
PROBE_N = 20000          # target probe-set size after subsampling
EVAL_EPISODES = 50       # eval episodes per (method, seed) for probe collection
PROBE_RNG_SEED = 42      # fixed RNG for reproducible subsampling
PROBE_DIR = Path("private/Metrics/Transfer/shared_probe")
RATE_DIR = Path("private/Metrics/Transfer")
CHECKPOINT_CACHE = Path("private/Checkpoints/transfer_wandb")


# ── W&B helpers ───────────────────────────────────────────────────────────────

def _find_wandb_runs(api, project: str, spec: dict) -> list:
    """Find W&B runs matching a method spec."""
    import wandb  # noqa
    runs = api.runs(project, filters=spec["wandb_filter"], per_page=100)
    matched = []
    for r in runs:
        tags = set(r.tags)
        if not all(t in tags for t in spec["match_tags"]):
            continue
        cfg_match = True
        for k, v in spec.get("match_config", {}).items():
            parts = k.split(".")
            val = r.config
            for p in parts:
                val = val.get(p, {}) if isinstance(val, dict) else None
            if val is None or abs(float(val) - float(v)) > 1e-8:
                cfg_match = False
                break
        if cfg_match:
            matched.append(r)
    return matched


def _download_checkpoint(run, cache_dir: Path) -> Path | None:
    """Download the best checkpoint from a W&B run to cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{run.name}.pt"
    if local_path.exists():
        print(f"      [cached] {local_path}")
        return local_path

    # Find checkpoint: prefer best-checkpoint*, fall back to checkpoint.pt
    candidates = []
    for f in run.files():
        if f.name.endswith(".pt"):
            if f.name.startswith("best-checkpoint"):
                candidates.insert(0, f)  # prefer best
            elif f.name == "checkpoint.pt":
                candidates.append(f)

    if candidates:
        chosen = candidates[0]
        print(f"      downloading {chosen.name} ({chosen.size / 1e6:.1f} MB) ...")
        with tempfile.TemporaryDirectory() as tmpdir:
            chosen.download(root=tmpdir, replace=True)
            downloaded = Path(tmpdir) / chosen.name
            downloaded.rename(local_path)
        return local_path

    print(f"      [WARN] no checkpoint file found for {run.name}")
    return None


def _save_wandb_config(run, cache_dir: Path) -> Path | None:
    """Save the W&B run's config dict as JSON (avoids YAML parsing issues)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{run.name}_config.json"
    if local_path.exists():
        return local_path
    try:
        cfg = dict(run.config)
        with open(local_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        return local_path
    except Exception as e:
        print(f"      [WARN] failed to save config for {run.name}: {e}")
        return None


# ── Agent loading ─────────────────────────────────────────────────────────────

def load_agent_from_wandb(checkpoint_path: Path, config_json_path: Path,
                          device: str) -> tuple[DCMPC | None, OmegaConf | None]:
    """Load a DCMPC agent from a W&B-downloaded checkpoint + config JSON.

    Returns (agent, train_cfg) so the caller can use train_cfg for env creation.
    """
    try:
        with open(config_json_path) as f:
            raw_cfg = json.load(f)
        train_cfg = OmegaConf.create(raw_cfg)
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
        env.close()

        agent = DCMPC(train_cfg.agent, obs_spec=obs_spec, act_spec=act_spec).to(device)
        state_dict = torch.load(
            str(checkpoint_path), weights_only=True, map_location=torch.device(device)
        )
        if "model" in state_dict:
            state_dict = state_dict["model"]
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent, train_cfg
    except Exception as e:
        print(f"      [ERROR] loading agent: {e}")
        import traceback; traceback.print_exc()
        return None, None


# ── Eval rollout ──────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_eval_observations(agent: DCMPC, cfg, n_episodes: int,
                              device: str) -> torch.Tensor:
    """Roll out the trained policy for n_episodes, return all state observations.

    The DC-MPC env wrapper stores observations as a nested TensorDict:
    td["observation"] is a TensorDict with key "state" → [obs_dim].
    We extract td["observation"]["state"] and flatten over (num_envs, T).
    """
    from tensordict.nn import TensorDictModule

    env = make_env(cfg, num_envs=1)
    action_repeat = getattr(cfg, "action_repeat", 2)
    max_steps = getattr(cfg, "max_episode_steps", 500) // action_repeat

    policy_module = TensorDictModule(
        lambda obs, step_count: agent.select_action(
            obs, t0=(step_count == 0), eval_mode=True
        ),
        in_keys=["observation", "step_count"],
        out_keys=["action"],
    )

    all_obs = []
    for ep in range(n_episodes):
        td = env.rollout(max_steps=max_steps, policy=policy_module)
        # td["observation"] is a nested TensorDict with batch_size [num_envs, T]
        # The actual state vector is td["observation"]["state"] → [num_envs, T, obs_dim]
        obs_td = td["observation"]
        if "state" in obs_td.keys():
            states = obs_td["state"]  # [num_envs, T, obs_dim]
        else:
            # Fallback: try to get the raw tensor
            states = obs_td
            if hasattr(states, "to_tensordict"):
                states = torch.cat([v for v in states.values()], dim=-1)

        # Flatten to [N, obs_dim]
        if states.dim() > 2:
            states = states.reshape(-1, states.shape[-1])
        elif states.dim() == 1:
            states = states.unsqueeze(0)
        all_obs.append(states.cpu())
    env.close()
    return torch.cat(all_obs, dim=0)


# ── Rate computation (reused from toy script) ────────────────────────────────

@torch.no_grad()
def compute_realized_rate(agent: DCMPC, probe_td: TensorDict) -> dict:
    """Encode probe set with eps=0, compute per-group Shannon entropy."""
    if not agent.model._uses_discrete:
        return {"bits_per_transition": float("nan"), "has_discrete": False}

    z_dict = agent.model.encode(probe_td, stochastic=False)
    indices = z_dict["indices"]
    if indices.ndim == 1:
        indices = indices.unsqueeze(-1)

    n_groups = indices.shape[-1]
    codebook_size = agent.model._quantizer.codebook_size

    total_H = 0.0
    for g in range(n_groups):
        group_tokens = indices[:, g].reshape(-1).to(torch.long)
        counts = torch.bincount(group_tokens, minlength=codebook_size)
        probs = counts.float() / counts.sum().clamp_min(1)
        probs = probs[probs > 0]
        total_H += float(-(probs * probs.log2()).sum().item())

    nominal_bits = n_groups * math.log2(codebook_size)
    return {
        "bits_per_transition": float(total_H),
        "bits_per_group": float(total_H) / n_groups,
        "codebook_size": int(codebook_size),
        "n_groups": int(n_groups),
        "nominal_bits": float(nominal_bits),
        "compression_ratio": float(nominal_bits / max(total_H, 1e-9)),
        "has_discrete": True,
    }


def _mean_ci95(values: list[float]) -> dict:
    """n/mean/std/95% t-CI for finite values."""
    vals = [float(v) for v in values if np.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    if n == 1:
        return {"n": 1, "mean": vals[0], "std": 0.0, "ci95": 0.0}
    arr = np.array(vals, dtype=float)
    tcrit = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n - 1, 1.96)
    std = float(arr.std(ddof=1))
    return {"n": n, "mean": float(arr.mean()), "std": std,
            "ci95": float(tcrit * std / math.sqrt(n))}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, help="Transfer env key (e.g. mw-button-press)")
    parser.add_argument("--wandb_project", default="f20210966/dcmpc")
    parser.add_argument("--probe_only", action="store_true",
                        help="Skip probe building; assume D*_task already exists")
    parser.add_argument("--build_only", action="store_true",
                        help="Only build the probe set; don't measure rate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval_episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--probe_n", type=int, default=PROBE_N)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.chdir(_ROOT)

    env_key = args.env
    if env_key not in TRANSFER_METHODS:
        print(f"[ERROR] Unknown env '{env_key}'. Available: {list(TRANSFER_METHODS.keys())}")
        return 1

    method_specs = TRANSFER_METHODS[env_key]
    probe_path = PROBE_DIR / f"D_star_{env_key}.pt"
    cache_dir = CHECKPOINT_CACHE / env_key

    import wandb
    api = wandb.Api()

    # ── Phase 1: Build probe set ──────────────────────────────────────────────
    if not args.probe_only:
        print(f"\n{'='*70}")
        print(f"Phase 1: Building shared probe set D*_{env_key}")
        print(f"{'='*70}")

        all_method_obs = []
        all_run_info = {}  # for provenance

        for spec in method_specs:
            print(f"\n  [{spec['label']}] Finding W&B runs ...")
            runs = _find_wandb_runs(api, args.wandb_project, spec)
            print(f"    Found {len(runs)} matching runs")

            run_obs_list = []
            for r in runs:
                print(f"    Run: {r.name}")
                ckpt_path = _download_checkpoint(r, cache_dir)
                cfg_path = _save_wandb_config(r, cache_dir)
                if ckpt_path is None or cfg_path is None:
                    print(f"      [SKIP] missing checkpoint or config")
                    continue

                agent, train_cfg = load_agent_from_wandb(ckpt_path, cfg_path, args.device)
                if agent is None or train_cfg is None:
                    continue

                print(f"      Rolling out {args.eval_episodes} eval episodes ...")
                obs = collect_eval_observations(agent, train_cfg,
                                                args.eval_episodes, args.device)
                print(f"      Collected {obs.shape[0]} observations (dim={obs.shape[1]})")
                run_obs_list.append(obs)
                del agent

            if run_obs_list:
                method_obs = torch.cat(run_obs_list, dim=0)
                all_method_obs.append(method_obs)
                all_run_info[spec["key"]] = {
                    "label": spec["label"],
                    "n_runs": len(run_obs_list),
                    "n_obs": int(method_obs.shape[0]),
                }
                print(f"    Total for {spec['label']}: {method_obs.shape[0]} obs")

        if not all_method_obs:
            print("[ERROR] No observations collected from any method")
            return 1

        # Pool and subsample
        pooled = torch.cat(all_method_obs, dim=0)
        print(f"\n  Pooled: {pooled.shape[0]} observations from {len(all_method_obs)} methods")

        rng = np.random.RandomState(PROBE_RNG_SEED)
        if pooled.shape[0] > args.probe_n:
            idx = rng.choice(pooled.shape[0], size=args.probe_n, replace=False)
            probe_obs = pooled[idx]
        else:
            probe_obs = pooled
            print(f"  [NOTE] pooled size ({pooled.shape[0]}) <= target N ({args.probe_n}); using all")

        print(f"  D*_{env_key}: {probe_obs.shape[0]} observations, dim={probe_obs.shape[1]}")

        # Save
        PROBE_DIR.mkdir(parents=True, exist_ok=True)
        probe_data = {
            "observations": probe_obs,
            "meta": {
                "env": env_key,
                "n_obs": int(probe_obs.shape[0]),
                "obs_dim": int(probe_obs.shape[1]),
                "eval_episodes_per_run": args.eval_episodes,
                "subsample_seed": PROBE_RNG_SEED,
                "target_n": args.probe_n,
                "pooled_from": all_run_info,
                "date": datetime.now().isoformat(),
                "description": (
                    f"Shared probe set for {env_key} transfer R_m. "
                    f"Pooled trained-policy eval rollouts from all compared methods, "
                    f"subsampled to {probe_obs.shape[0]} with seed {PROBE_RNG_SEED}. "
                    f"Use identical D* for all methods to remove T6 visitation confound."
                ),
            },
        }
        torch.save(probe_data, probe_path)
        print(f"  Saved: {probe_path}")

        if args.build_only:
            print("\n  --build_only: stopping after probe construction.")
            return 0

    # ── Phase 2: Measure R_m ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Phase 2: Measuring R_m on D*_{env_key}")
    print(f"{'='*70}")

    if not probe_path.exists():
        print(f"[ERROR] Probe set not found at {probe_path}. Run without --probe_only first.")
        return 1

    probe_data = torch.load(probe_path, weights_only=False)
    probe_obs = probe_data["observations"]
    print(f"  Loaded D*: {probe_obs.shape[0]} obs, dim={probe_obs.shape[1]}")

    probe_td = TensorDict({"state": probe_obs}, batch_size=[probe_obs.shape[0]])

    results = {
        "meta": {
            "env": env_key,
            "probe_path": str(probe_path),
            "probe_n": int(probe_obs.shape[0]),
            "probe_obs_dim": int(probe_obs.shape[1]),
            "rate_metric": f"R_m(D*_{env_key}) = sum_g H[codes_g(s)], s ~ shared pooled probe, eps=0",
            "date": datetime.now().isoformat(),
        },
        "methods": {},
    }

    for spec in method_specs:
        print(f"\n  [{spec['label']}]")
        runs = _find_wandb_runs(api, args.wandb_project, spec)
        seed_rates = []

        for r in runs:
            ckpt_path = cache_dir / f"{r.name}.pt"
            cfg_path = cache_dir / f"{r.name}_config.json"
            if not ckpt_path.exists() or not cfg_path.exists():
                print(f"    [SKIP] {r.name} — checkpoint/config not cached")
                continue

            agent, _ = load_agent_from_wandb(ckpt_path, cfg_path, args.device)
            if agent is None:
                continue

            rate = compute_realized_rate(agent, probe_td)
            if rate["has_discrete"]:
                print(f"    {r.name}: R_m = {rate['bits_per_transition']:.2f} bits/trans")
                seed_rates.append({
                    "run_name": r.name,
                    "rate": rate,
                })
            else:
                print(f"    {r.name}: continuous (no quantizer)")
            del agent

        summary = _mean_ci95([s["rate"]["bits_per_transition"] for s in seed_rates])
        print(f"    Summary: R_m = {summary['mean']:.2f} ± {summary['ci95']:.2f} "
              f"(n={summary['n']})")

        results["methods"][spec["key"]] = {
            "label": spec["label"],
            "seeds": seed_rates,
            "summary": {
                "bits_per_transition": summary,
            },
        }

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Method':<25} {'R_m (bits)':>12} {'±CI95':>8} {'n':>4}")
    print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*4}")
    for key, entry in results["methods"].items():
        s = entry["summary"]["bits_per_transition"]
        print(f"  {entry['label']:<25} {s['mean']:>12.2f} {s['ci95']:>8.2f} {s['n']:>4}")

    # ── Save ──────────────────────────────────────────────────────────────────
    RATE_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = Path(args.output) if args.output else RATE_DIR / f"e7_transfer_Rm_{env_key}_{date_str}.json"

    # Convert non-serializable items
    save_results = json.loads(json.dumps(results, default=str))
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
