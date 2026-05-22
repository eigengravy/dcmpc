#!/usr/bin/env bash
set -euo pipefail

# Remote-cluster dispatcher for non-toy workshop-paper runs.
# Intended for collaborators to launch on a Linux GPU cluster, not locally.
#
# Stages:
#   toy_pareto
#   metaworld_baselines
#   metaworld_ddcl_sensitivity
#   dmcontrol_baselines

STAGE="${STAGE:-help}"
LAUNCHER="${LAUNCHER:-slurm}"
DEVICE="${DEVICE:-cuda}"
GRES="${GRES:-gpu:1}"
WANDB_METAWORLD_PROJECT_NAME="${WANDB_METAWORLD_PROJECT_NAME:-ddcl_mbrl_metaworld}"
WANDB_DMCONTROL_PROJECT_NAME="${WANDB_DMCONTROL_PROJECT_NAME:-ddcl_mbrl_dmcontrol}"
WANDB_TOY_PROJECT_NAME="${WANDB_TOY_PROJECT_NAME:-ddcl_mbrl_toy_pareto}"

case "${STAGE}" in
  toy_pareto)
    LAUNCHER="${LAUNCHER}" \
    DEVICE="${DEVICE}" \
    GRES="${GRES}" \
    WANDB_PROJECT_NAME="${WANDB_TOY_PROJECT_NAME}" \
    scripts/slurm_toy_quantizer_pareto.sh
    ;;

  metaworld_baselines)
    LAUNCHER="${LAUNCHER}" \
    DEVICE="${DEVICE}" \
    GRES="${GRES}" \
    WANDB_PROJECT_NAME="${WANDB_METAWORLD_PROJECT_NAME}" \
    scripts/slurm_metaworld_button_press_baselines.sh
    ;;

  metaworld_ddcl_sensitivity)
    LAUNCHER="${LAUNCHER}" \
    DEVICE="${DEVICE}" \
    GRES="${GRES}" \
    WANDB_PROJECT_NAME="${WANDB_METAWORLD_PROJECT_NAME}" \
    DDCL_AGENT="${DDCL_AGENT:-ddcl_cosine}" \
    scripts/slurm_metaworld_ddcl_sensitivity.sh
    ;;

  dmcontrol_baselines)
    LAUNCHER="${LAUNCHER}" \
    DEVICE="${DEVICE}" \
    GRES="${GRES}" \
    WANDB_PROJECT_NAME="${WANDB_DMCONTROL_PROJECT_NAME}" \
    scripts/slurm_dmcontrol_baselines.sh
    ;;

  help|*)
    cat <<'EOF'
Usage on the remote cluster:
  STAGE=toy_pareto scripts/reproduce_remote_paper_runs.sh
  STAGE=metaworld_baselines scripts/reproduce_remote_paper_runs.sh
  STAGE=metaworld_ddcl_sensitivity scripts/reproduce_remote_paper_runs.sh
  STAGE=dmcontrol_baselines scripts/reproduce_remote_paper_runs.sh

Useful overrides:
  LAUNCHER=slurm
  DEVICE=cuda
  GRES=gpu:a100:1
  WANDB_METAWORLD_PROJECT_NAME=ddcl_mbrl_metaworld
  WANDB_DMCONTROL_PROJECT_NAME=ddcl_mbrl_dmcontrol

Cluster jobs should keep restore_best_checkpoint_at_end=true and
log_best_checkpoint_eval=true. train.py uploads the selected checkpoint.pt to
W&B as the eval-checkpoint model artifact for future eval.py re-runs.
EOF
    ;;
esac
