#!/usr/bin/env python3
"""
Stochasticity Ablation Pipeline — complete all remaining training, eval, and extraction.

Phase 1: Launch the 15 remaining training runs (ddcl_ce_stoch_eval × 5 seeds,
         ddcl_ce_stoch_tgt × 5 seeds, ddcl_cos_stoch_eval × 5 seeds) into the
         existing stoch ablation run directory.

Phase 2: Wait for ALL 20 training runs to complete, including the 5 already-running
         ddcl_ce_stoch seeds (tracked via kill-0 polling on their PIDs).

Phase 3: Run heldout eval (50 episodes, seed=10000) for all 4 conditions × 5 seeds.

Phase 4: Extract per-seed metrics from local W&B summary JSONs and write the
         canonical heldout eval JSON to private/Metrics/Toy/Corrected Reevaluation/.

Usage (from dcmpc/ directory, with conda env active):
    python scripts/run_stoch_ablation_pipeline.py

Environment variables:
    SKIP_TRAIN=1      — skip Phase 1–2 (training), go straight to eval
    SKIP_EVAL=1       — skip Phase 3 (eval), go straight to extraction
    EVAL_ROOT=<path>  — override the eval output directory (Phase 3)
    MAX_TRAIN_PAR=N   — max training jobs in parallel beyond the 5 already running (default 3)
    MAX_EVAL_PAR=N    — max eval jobs in parallel (default 5)
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

# Absolute paths (script can be run from any directory)
SCRIPT_DIR = Path(__file__).parent
DCMPC_DIR  = SCRIPT_DIR.parent  # dcmpc/

STOCH_RUN_ROOT = DCMPC_DIR / "output" / "toy_runs" / "ddcl_stoch_ablation_20260519_210503"

METRICS_OUT_DIR = (
    DCMPC_DIR / "private" / "Metrics" / "Toy" / "Corrected Reevaluation"
)

ENV_NAME = "toy-precision-gate-final"

# The 4 ablation conditions
REMAINING_TRAIN_AGENTS = [
    "ddcl_ce_stoch_eval",   # DDCL-CE(s,d): stoch eval + det targets
    "ddcl_ce_stoch_tgt",    # DDCL-CE(d,s): det eval  + stoch targets
    "ddcl_cos_stoch_eval",  # DDCL-Cos(s,d): stoch eval + det targets
]
ALL_TRAIN_AGENTS = ["ddcl_ce_stoch"] + REMAINING_TRAIN_AGENTS

SEEDS = [0, 1, 2, 3, 4]

# PIDs of the 5 ddcl_ce_stoch seeds that are currently running.
# If they have already finished by the time this script runs, kill-0 will
# immediately return "not running" and Phase 2 will skip them gracefully.
EXISTING_TRAIN_PIDS = [41681, 41670, 41671, 41690, 41673]

# Parallelism
MAX_TRAIN_PAR = int(os.environ.get("MAX_TRAIN_PAR", "3"))  # extra slots on top of existing 5
MAX_EVAL_PAR  = int(os.environ.get("MAX_EVAL_PAR",  "5"))

SKIP_TRAIN = os.environ.get("SKIP_TRAIN", "0") == "1"
SKIP_EVAL  = os.environ.get("SKIP_EVAL",  "0") == "1"

WANDB_TRAIN_PROJECT = "ddcl_mbrl_toy"
WANDB_EVAL_PROJECT  = "ddcl_mbrl_toy_eval"
WANDB_TRAIN_TAGS    = "ddcl-stoch-ablation-v1"
NUM_EVAL_EPISODES   = 50
EVAL_SEED           = 10000

METRIC_REMAP = {
    "episodic_success":                          "episodic_success",
    "episodic_precision_success_0p02":            "episodic_precision_success_0p02",
    "episodic_precision_success_0p04":            "episodic_precision_success_0p04",
    "episodic_precision_success_0p08":            "episodic_precision_success_0p08",
    "episodic_return":                            "episodic_return",
    "rate/empirical_entropy_bits_per_transition": "rate/empirical_entropy_bits_per_transition",
    "rate/allocated_bits_per_transition":         "rate/allocated_bits_per_transition",
    "codebook/usage_percent":                     "codebook/usage_percent",
    "codebook/per_group_entropy_mean":            "codebook/per_group_entropy_mean",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_pid_running(pid: int) -> bool:
    """Return True if the process with the given PID is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we cannot signal it


def count_alive(procs: list[subprocess.Popen]) -> int:
    return sum(1 for p in procs if p.poll() is None)


def wait_for_existing_pids(pids: list[int], poll_interval: int = 60) -> None:
    """Poll until all external PIDs have exited."""
    remaining = {p for p in pids if is_pid_running(p)}
    if not remaining:
        print("  All existing training PIDs have already exited.")
        return
    print(f"  Waiting for existing PIDs to finish: {sorted(remaining)}")
    while remaining:
        time.sleep(poll_interval)
        remaining = {p for p in remaining if is_pid_running(p)}
        if remaining:
            print(f"  Still running: {sorted(remaining)}")
    print("  All existing training PIDs finished.")


def make_env_overrides() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        VECLIB_MAXIMUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
        HYDRA_FULL_ERROR="1",
    )
    return env


# ─── Phase 1: Training ────────────────────────────────────────────────────────

def launch_training(agent: str, seed: int) -> tuple[subprocess.Popen, object]:
    run_id   = f"{ENV_NAME}--{agent}--s{seed}"
    log_path = STOCH_RUN_ROOT / "logs" / f"{run_id}.log"
    hydra_dir = STOCH_RUN_ROOT / "hydra" / run_id

    # Skip if this run already exists (e.g. re-running after partial failure)
    if hydra_dir.exists():
        print(f"  SKIP (already exists): {run_id}")
        return None, None

    hydra_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(DCMPC_DIR / "train.py"),
        f"env={ENV_NAME}",
        f"agent={agent}",
        f"seed={seed}",
        "device=auto",
        "use_wandb=true",
        f"wandb_project_name={WANDB_TRAIN_PROJECT}",
        f"+wandb_tags=[{WANDB_TRAIN_TAGS}]",
        f"hydra.run.dir={hydra_dir}",
    ]
    log_file = open(log_path, "w")
    print(f"  Launching train: {run_id}")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=make_env_overrides(),
                            cwd=str(DCMPC_DIR))
    return proc, log_file


def run_phase1_training() -> None:
    print("\n" + "=" * 70)
    print("Phase 1: Launching remaining 15 training runs")
    print(f"  Run root: {STOCH_RUN_ROOT}")
    print(f"  Max additional parallel: {MAX_TRAIN_PAR}")
    print("=" * 70)

    procs: list[subprocess.Popen] = []
    log_files = []

    for agent in REMAINING_TRAIN_AGENTS:
        for seed in SEEDS:
            hydra_dir = STOCH_RUN_ROOT / "hydra" / f"{ENV_NAME}--{agent}--s{seed}"
            if hydra_dir.exists():
                print(f"  SKIP (already exists): {agent} s{seed}")
                continue
            # Wait until we have a free slot (don't count finished processes)
            while count_alive(procs) >= MAX_TRAIN_PAR:
                time.sleep(30)
            proc, log_file = launch_training(agent, seed)
            if proc is not None:
                procs.append(proc)
                log_files.append(log_file)

    print(f"\n  Launched {len(procs)} new training processes. Waiting for them...")
    for proc, lf in zip(procs, log_files):
        ret = proc.wait()
        if lf:
            lf.close()
        if ret != 0:
            print(f"  WARNING: training process PID {proc.pid} exited with code {ret}")

    print("  New training runs finished (or were skipped).")


# ─── Phase 2: Wait for all 20 training runs ───────────────────────────────────

def run_phase2_wait() -> None:
    print("\n" + "=" * 70)
    print("Phase 2: Waiting for all 20 training runs to complete")
    print("=" * 70)
    wait_for_existing_pids(EXISTING_TRAIN_PIDS, poll_interval=60)

    # Verify all checkpoints exist
    missing = []
    for agent in ALL_TRAIN_AGENTS:
        for seed in SEEDS:
            ckpt = STOCH_RUN_ROOT / "hydra" / f"{ENV_NAME}--{agent}--s{seed}" / "checkpoint.pt"
            if not ckpt.exists():
                missing.append(f"{agent}--s{seed}")

    if missing:
        print(f"\n  WARNING: {len(missing)} checkpoints still missing:")
        for m in missing:
            print(f"    {m}")
        print("  These seeds will be skipped in the eval phase.")
    else:
        print(f"  All {len(ALL_TRAIN_AGENTS) * len(SEEDS)} checkpoints found.")


# ─── Phase 3: Heldout eval ────────────────────────────────────────────────────

def launch_eval(agent: str, seed: int, checkpoint: Path, eval_run_root: Path) -> tuple[subprocess.Popen, object]:
    run_id    = f"stoch_ablation_heldout--{ENV_NAME}--{agent}--s{seed}"
    log_path  = eval_run_root / "logs" / f"{run_id}.log"
    hydra_dir = eval_run_root / "hydra" / run_id

    hydra_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(DCMPC_DIR / "eval.py"),
        f"checkpoint={checkpoint}",
        "device=auto",
        "use_wandb=true",
        f"wandb_project_name={WANDB_EVAL_PROJECT}",
        f"run_name=paper-heldout-{run_id}",
        f"num_eval_episodes={NUM_EVAL_EPISODES}",
        f"eval_seed={EVAL_SEED}",
        "capture_eval_video=false",
        f"hydra.run.dir={hydra_dir}",
    ]
    log_file = open(log_path, "w")
    print(f"  Launching eval: {run_id}")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=make_env_overrides(),
                            cwd=str(DCMPC_DIR))
    return proc, log_file


def run_phase3_eval(eval_run_root: Path) -> None:
    print("\n" + "=" * 70)
    print("Phase 3: Heldout eval (50 eps, seed=10000) for all 4 × 5 seeds")
    print(f"  Eval root: {eval_run_root}")
    print(f"  Max parallel: {MAX_EVAL_PAR}")
    print("=" * 70)

    eval_run_root.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    log_files = []

    for agent in ALL_TRAIN_AGENTS:
        for seed in SEEDS:
            ckpt = STOCH_RUN_ROOT / "hydra" / f"{ENV_NAME}--{agent}--s{seed}" / "checkpoint.pt"
            if not ckpt.exists():
                print(f"  SKIP (no checkpoint): {agent} s{seed}")
                continue

            # Skip if already evaluated
            hydra_dir = eval_run_root / "hydra" / f"stoch_ablation_heldout--{ENV_NAME}--{agent}--s{seed}"
            if hydra_dir.exists():
                print(f"  SKIP (already evaluated): {agent} s{seed}")
                continue

            while count_alive(procs) >= MAX_EVAL_PAR:
                time.sleep(30)

            proc, log_file = launch_eval(agent, seed, ckpt, eval_run_root)
            procs.append(proc)
            log_files.append(log_file)

    print(f"\n  Launched {len(procs)} eval jobs. Waiting...")
    failed = 0
    for proc, lf in zip(procs, log_files):
        ret = proc.wait()
        if lf:
            lf.close()
        if ret != 0:
            print(f"  WARNING: eval PID {proc.pid} exited with code {ret}")
            failed += 1

    if failed:
        print(f"\n  {failed} eval job(s) failed. Check {eval_run_root}/logs/")
        sys.exit(1)

    print(f"  All {len(procs)} eval jobs complete.")


# ─── Phase 4: Extract metrics ─────────────────────────────────────────────────

def load_summary_metrics(eval_root: Path, agent: str, seed: int) -> dict[str, float]:
    """Load per-seed metrics from local wandb-summary.json for one eval run."""
    run_id = f"stoch_ablation_heldout--{ENV_NAME}--{agent}--s{seed}"
    hydra_dir = eval_root / "hydra" / run_id
    wandb_dir = hydra_dir / "wandb"
    if not wandb_dir.exists():
        return {}

    run_dirs = sorted(wandb_dir.glob("run-*"))
    if not run_dirs:
        return {}
    summary_path = run_dirs[-1] / "files" / "wandb-summary.json"
    if not summary_path.exists():
        return {}

    with open(summary_path) as f:
        summary = json.load(f)

    eval_block = summary.get("eval/", summary.get("eval", {}))
    if not isinstance(eval_block, dict):
        eval_block = {}

    flat = {k: v for k, v in summary.items() if isinstance(v, (int, float))}

    out: dict[str, float] = {}
    for wb_key, canon_key in METRIC_REMAP.items():
        val = eval_block.get(wb_key)
        if val is None:
            val = flat.get(wb_key)
        if val is None:
            val = flat.get(f"eval/{wb_key}")
        if val is not None:
            out[canon_key] = float(val)

    return out


def compute_stats(values: list[float]) -> dict:
    import math
    import statistics

    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "values": []}
    mean = statistics.mean(values)
    std  = statistics.pstdev(values) if n == 1 else statistics.stdev(values)
    if n >= 2:
        # t_{n-1, 0.975} approximation via scipy if available, else use table
        try:
            from scipy import stats as scipy_stats
            t_val = scipy_stats.t.ppf(0.975, df=n - 1)
        except ImportError:
            t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                       7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
            t_val = t_table.get(n, 1.96)
        ci95 = t_val * std / math.sqrt(n)
    else:
        ci95 = float("nan")
    return {"n": n, "mean": mean, "std": std, "ci95": ci95, "values": values}


def run_phase4_extract(eval_run_root: Path) -> None:
    print("\n" + "=" * 70)
    print("Phase 4: Extracting metrics from local W&B summary files")
    print(f"  Eval root: {eval_run_root}")
    print("=" * 70)

    output: dict = {}

    for agent in ALL_TRAIN_AGENTS:
        per_seed: dict[str, list[float]] = {v: [] for v in METRIC_REMAP.values()}
        run_ids: list[str] = []
        seeds_found: list[int] = []

        print(f"\n  Agent: {agent}")
        for seed in SEEDS:
            metrics = load_summary_metrics(eval_run_root, agent, seed)
            if not metrics:
                print(f"    seed {seed}: NO DATA (missing or not yet evaluated)")
                continue
            seeds_found.append(seed)
            run_ids.append(f"s{seed}")
            for canon_key in METRIC_REMAP.values():
                val = metrics.get(canon_key)
                if val is not None:
                    per_seed[canon_key].append(val)
            print(f"    seed {seed}: success={metrics.get('episodic_success', 'N/A'):.3f}  "
                  f"prec@02={metrics.get('episodic_precision_success_0p02', 'N/A'):.3f}")

        metrics_blob: dict = {}
        for canon_key, values in per_seed.items():
            if not values:
                continue
            st = compute_stats(values)
            metrics_blob[canon_key] = {
                "n": st["n"], "mean": st["mean"], "std": st["std"],
                "ci95": st["ci95"], "values": st["values"],
            }

        output[agent] = {
            "ids": run_ids,
            "name_count": len(run_ids),
            "metrics": metrics_blob,
        }

        if "episodic_success" in metrics_blob:
            d = metrics_blob["episodic_success"]
            p = metrics_blob.get("episodic_precision_success_0p02", {})
            print(f"    → success  : {d['mean']:.3f} ± {d['ci95']:.3f}  (n={d['n']})")
            if p:
                print(f"    → prec@0.02: {p['mean']:.3f} ± {p['ci95']:.3f}  (n={p['n']})")

    # Save canonical JSON
    METRICS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = METRICS_OUT_DIR / f"wandb_toy_stoch_ablation_5seed_heldout_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Summary CI table
    print("\n── Stochasticity ablation 95% CI summary ─────────────────────────")
    labels = {
        "ddcl_ce_stoch":      "DDCL-CE(s,s)  [stoch eval + stoch tgt]",
        "ddcl_ce_stoch_eval": "DDCL-CE(s,d)  [stoch eval + det   tgt]",
        "ddcl_ce_stoch_tgt":  "DDCL-CE(d,s)  [det   eval + stoch tgt]",
        "ddcl_cos_stoch_eval":"DDCL-Cos(s,d) [stoch eval + det   tgt]",
    }
    for agent in ALL_TRAIN_AGENTS:
        lbl = labels.get(agent, agent)
        m = output.get(agent, {}).get("metrics", {})
        s = m.get("episodic_success", {})
        p = m.get("episodic_precision_success_0p02", {})
        if s:
            print(f"  {lbl}")
            print(f"    success  = {s['mean']:.3f} ± {s['ci95']:.3f}  (n={s['n']}, "
                  f"per-seed={[f'{v:.3f}' for v in s['values']]})")
            if p:
                print(f"    prec@0.02= {p['mean']:.3f} ± {p['ci95']:.3f}  (n={p['n']})")
        else:
            print(f"  {lbl}: NO DATA")

    print(f"\nDone. Output: {out_path}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.chdir(DCMPC_DIR)  # ensure relative paths work

    # Determine eval root (fixed after first run to allow reuse across invocations)
    eval_root_override = os.environ.get("EVAL_ROOT", "")
    if eval_root_override:
        eval_run_root = Path(eval_root_override)
    else:
        # Use a fixed name so SKIP_EVAL=1 can find the existing eval dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_run_root = DCMPC_DIR / "output" / "toy_runs" / f"stoch_ablation_heldout_eval_{ts}"

    print("=" * 70)
    print("DDCL Stochasticity Ablation Pipeline")
    print(f"  Training root : {STOCH_RUN_ROOT}")
    print(f"  Eval root     : {eval_run_root}")
    print(f"  Metrics out   : {METRICS_OUT_DIR}")
    print(f"  Skip train    : {SKIP_TRAIN}")
    print(f"  Skip eval     : {SKIP_EVAL}")
    print("=" * 70)

    if not SKIP_TRAIN:
        run_phase1_training()
        run_phase2_wait()
    else:
        print("\nPhases 1–2 skipped (SKIP_TRAIN=1).")
        # Still verify checkpoints
        run_phase2_wait()

    if not SKIP_EVAL:
        run_phase3_eval(eval_run_root)
    else:
        # Try to find existing eval root
        existing = sorted(
            (DCMPC_DIR / "output" / "toy_runs").glob("stoch_ablation_heldout_eval_*")
        )
        if existing:
            eval_run_root = existing[-1]
            print(f"\nPhase 3 skipped (SKIP_EVAL=1). Using existing eval root: {eval_run_root}")
        else:
            print("\nPhase 3 skipped but no existing eval root found. Extraction may fail.")

    run_phase4_extract(eval_run_root)


if __name__ == "__main__":
    main()
