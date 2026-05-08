#!/usr/bin/env bash
set -euo pipefail

LAUNCHER="${LAUNCHER:-slurm}"
ENV_NAME="${ENV_NAME:-mw-button-press}"
SEEDS="${SEEDS:-0,1,2,3,4}"
AGENTS="${AGENTS:-ddcl_ce,dcmpc,vq_ce,continuous_mse}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_metaworld}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1440}"
MEM_GB="${MEM_GB:-32}"
CPUS_PER_TASK="${CPUS_PER_TASK:-5}"
GRES="${GRES:-gpu:1}"

python train.py --multirun \
  "hydra/launcher=${LAUNCHER}" \
  "env=${ENV_NAME}" \
  "agent=${AGENTS}" \
  "seed=${SEEDS}" \
  "device=${DEVICE}" \
  "use_wandb=true" \
  "wandb_project_name=${WANDB_PROJECT_NAME}" \
  "hydra.launcher.timeout_min=${TIMEOUT_MIN}" \
  "hydra.launcher.mem_gb=${MEM_GB}" \
  "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}" \
  "hydra.launcher.gres=${GRES}"
