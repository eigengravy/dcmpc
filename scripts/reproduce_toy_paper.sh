#!/usr/bin/env bash
set -euo pipefail

# Reproduce the toy experiments used by the workshop-paper workflow.
# This script is a dispatcher; set STAGE to choose the run family.
#
# Stages:
#   baselines              final toy baselines, 5 seeds
#   ddcl_ce_repair         DDCL CE repair/stabilization sweep
#   ddcl_objectives        original DDCL MSE/cosine objective sweep, seeds 0-2
#   ddcl_objective_finish  selected DDCL MSE/cosine seeds 3-4
#   pareto                 quantizer Pareto sweep over FSQ/VQ size and DDCL lambda
#   reeval_baselines       eval-only baseline/CE checkpoint re-evaluation
#   reeval_objectives      eval-only MSE/cosine checkpoint re-evaluation
#   plots                  build paper plots from saved aggregate JSON files

STAGE="${STAGE:-help}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
WANDB_EVAL_PROJECT_NAME="${WANDB_EVAL_PROJECT_NAME:-ddcl_mbrl_toy_eval}"
TOY_METRICS_DIR="${TOY_METRICS_DIR:-private/Metrics/Toy/Corrected Reevaluation}"

case "${STAGE}" in
  baselines)
    ENV_NAME=toy-precision-gate-final \
    AGENTS="ddcl_ce dcmpc vq_ce continuous_mse" \
    SEEDS="0 1 2 3 4" \
    MAX_PARALLEL="${MAX_PARALLEL:-4}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DEVICE="${DEVICE}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME}" \
    scripts/run_toy_baselines_parallel.sh
    ;;

  ddcl_ce_repair)
    PYTHON_BIN="${PYTHON_BIN}" \
    DEVICE="${DEVICE}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME}" \
    scripts/run_toy_ddcl_ce_repair_sweep.sh
    ;;

  ddcl_objectives)
    PYTHON_BIN="${PYTHON_BIN}" \
    DEVICE="${DEVICE}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME}" \
    scripts/run_toy_ddcl_objective_sweep.sh
    ;;

  ddcl_objective_finish)
    PYTHON_BIN="${PYTHON_BIN}" \
    DEVICE="${DEVICE}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME}" \
    scripts/run_toy_ddcl_objective_completion.sh
    ;;

  pareto)
    PYTHON_BIN="${PYTHON_BIN}" \
    DEVICE="${DEVICE}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy_pareto}" \
    scripts/run_toy_quantizer_pareto_sweep.sh
    ;;

  reeval_baselines)
    PYTHON_BIN="${PYTHON_BIN}" \
    DEVICE="${DEVICE}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_PROJECT_NAME="${WANDB_EVAL_PROJECT_NAME}" \
    scripts/reevaluate_toy_paper_checkpoints.sh
    ;;

  reeval_objectives)
    PYTHON_BIN="${PYTHON_BIN}" \
    DEVICE="${DEVICE}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_PROJECT_NAME="${WANDB_EVAL_PROJECT_NAME}" \
    scripts/reevaluate_toy_objective_checkpoints.sh
    ;;

  plots)
    "${PYTHON_BIN}" results/plotting/make_paper_plots.py \
      --input "${TOY_METRICS_DIR}/wandb_toy_corrected_reeval_20260515.json" \
      --input "${TOY_METRICS_DIR}/wandb_toy_objective_corrected_reeval_20260515.json" \
      --outdir results/paper_plots/toy
    ;;

  help|*)
    cat <<'EOF'
Usage:
  STAGE=baselines scripts/reproduce_toy_paper.sh
  STAGE=ddcl_ce_repair scripts/reproduce_toy_paper.sh
  STAGE=ddcl_objectives scripts/reproduce_toy_paper.sh
  STAGE=ddcl_objective_finish scripts/reproduce_toy_paper.sh
  STAGE=pareto scripts/reproduce_toy_paper.sh
  STAGE=reeval_baselines scripts/reproduce_toy_paper.sh
  STAGE=reeval_objectives scripts/reproduce_toy_paper.sh
  STAGE=plots scripts/reproduce_toy_paper.sh

Set PYTHON_BIN, DEVICE, USE_WANDB, WANDB_PROJECT_NAME, TOY_METRICS_DIR, and MAX_PARALLEL as needed.
EOF
    ;;
esac
