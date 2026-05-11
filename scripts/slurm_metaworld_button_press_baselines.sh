#!/usr/bin/env bash
set -euo pipefail

LAUNCHER="${LAUNCHER:-slurm}"
ENV_NAME="${ENV_NAME:-mw-button-press}"
SEEDS="${SEEDS:-0,1,2,3,4}"
AGENTS="${AGENTS:-ddcl_ce,dcmpc,vq_ce,continuous_mse}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_metaworld}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-metaworld-paper-v1}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}"
RESTORE_BEST_CHECKPOINT="${RESTORE_BEST_CHECKPOINT:-true}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${SWEEP_DIR:-output/metaworld_runs/${ENV_NAME}_baselines_${RUN_ID}/hydra}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1440}"
MEM_GB="${MEM_GB:-32}"
CPUS_PER_TASK="${CPUS_PER_TASK:-5}"
GRES="${GRES:-gpu:1}"

echo "Submitting Meta-World baselines"
echo "  env=${ENV_NAME}"
echo "  agents=${AGENTS}"
echo "  seeds=${SEEDS}"
echo "  sweep_dir=${SWEEP_DIR}"
echo "  experiment_tag=${EXPERIMENT_TAG}"

python train.py --multirun \
  "hydra/launcher=${LAUNCHER}" \
  "env=${ENV_NAME}" \
  "agent=${AGENTS}" \
  "seed=${SEEDS}" \
  "device=${DEVICE}" \
  "use_wandb=true" \
  "wandb_project_name=${WANDB_PROJECT_NAME}" \
  "experiment_tag=${EXPERIMENT_TAG}" \
  "num_eval_episodes=${NUM_EVAL_EPISODES}" \
  "restore_best_checkpoint_at_end=${RESTORE_BEST_CHECKPOINT}" \
  "log_best_checkpoint_eval=true" \
  "hydra.sweep.dir=${SWEEP_DIR}" \
  "hydra.sweep.subdir=\${hydra.job.num}" \
  "hydra.launcher.timeout_min=${TIMEOUT_MIN}" \
  "hydra.launcher.mem_gb=${MEM_GB}" \
  "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}" \
  "hydra.launcher.gres=${GRES}"
