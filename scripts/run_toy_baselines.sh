#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-toy-precision-gate}"
SEEDS="${SEEDS:-0 1 2 3 4}"
AGENTS="${AGENTS:-ddcl_ce dcmpc vq_ce continuous_mse}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for agent in ${AGENTS}; do
  for seed in ${SEEDS}; do
    echo "Launching toy baseline: env=${ENV_NAME} agent=${agent} seed=${seed} project=${WANDB_PROJECT_NAME}"
    "${PYTHON_BIN}" train.py \
      env="${ENV_NAME}" \
      agent="${agent}" \
      seed="${seed}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}"
  done
done
