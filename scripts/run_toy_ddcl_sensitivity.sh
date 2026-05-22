#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-toy-precision-gate}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LAMBDAS="${LAMBDAS:-0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2}"
DELTAS="${DELTAS:-0.25 0.5 1.0 2.0}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for lambda in ${LAMBDAS}; do
  for delta in ${DELTAS}; do
    for seed in ${SEEDS}; do
      echo "Launching toy DDCL sensitivity: env=${ENV_NAME} lambda=${lambda} delta=${delta} seed=${seed} project=${WANDB_PROJECT_NAME}"
      "${PYTHON_BIN}" train.py \
        env="${ENV_NAME}" \
        agent=ddcl_ce \
        seed="${seed}" \
        device="${DEVICE}" \
        use_wandb="${USE_WANDB}" \
        wandb_project_name="${WANDB_PROJECT_NAME}" \
        agent.ddcl_lambda="${lambda}" \
        agent.ddcl_delta="${delta}"
    done
  done
done
