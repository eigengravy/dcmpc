#!/usr/bin/env bash
set -euo pipefail

LAUNCHER="${LAUNCHER:-slurm}"
ENV_NAME="${ENV_NAME:-mw-button-press}"
SEEDS="${SEEDS:-0,1,2,3,4}"
LAMBDAS="${LAMBDAS:-0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3}"
DELTAS="${DELTAS:-0.5,1.0,2.0}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_metaworld}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1440}"
MEM_GB="${MEM_GB:-32}"
CPUS_PER_TASK="${CPUS_PER_TASK:-5}"
GRES="${GRES:-gpu:1}"

python train.py --multirun \
  "hydra/launcher=${LAUNCHER}" \
  "env=${ENV_NAME}" \
  "agent=ddcl_ce" \
  "seed=${SEEDS}" \
  "device=${DEVICE}" \
  "use_wandb=true" \
  "wandb_project_name=${WANDB_PROJECT_NAME}" \
  "agent.ddcl_lambda=${LAMBDAS}" \
  "agent.ddcl_delta=${DELTAS}" \
  "hydra.launcher.timeout_min=${TIMEOUT_MIN}" \
  "hydra.launcher.mem_gb=${MEM_GB}" \
  "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}" \
  "hydra.launcher.gres=${GRES}"
