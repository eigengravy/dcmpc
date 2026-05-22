#!/usr/bin/env bash
set -euo pipefail

# Stochasticity ablation for DDCL toy experiments.
#
# Runs 4 conditions × 5 seeds to populate Table 2 in the paper:
#   - DDCL-CE(s,s): stoch eval + stoch targets  [ddcl_ce_stoch]
#   - DDCL-CE(s,d): stoch eval + det  targets   [ddcl_ce_stoch_eval]
#   - DDCL-CE(d,s): det  eval + stoch targets   [ddcl_ce_stoch_tgt]
#   - DDCL-Cos(s,d): stoch eval + det  targets  [ddcl_cos_stoch_eval]
#
# Existing DDCL-CE(d,d) and DDCL-Cos(d,d) are in the standard heldout JSON.
#
# Usage (from dcmpc/ directory):
#   conda run -n ddcl_mbrl bash scripts/run_toy_ddcl_stochasticity_ablation.sh

ENV_NAME="${ENV_NAME:-toy-precision-gate-final}"
AGENTS="${AGENTS:-ddcl_ce_stoch ddcl_ce_stoch_eval ddcl_ce_stoch_tgt ddcl_cos_stoch_eval}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
WANDB_TAGS="${WANDB_TAGS:-ddcl-stoch-ablation-v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
NICE_LEVEL="${NICE_LEVEL:-5}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/ddcl_stoch_ablation_$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${RUN_ROOT}/manifest.txt"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/hydra"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

pids=()

wait_for_slot() {
  local active_jobs
  active_jobs="$(jobs -pr | wc -l | tr -d '[:space:]')"
  active_jobs="${active_jobs:-0}"
  while (( active_jobs >= MAX_PARALLEL )); do
    sleep 15
    active_jobs="$(jobs -pr | wc -l | tr -d '[:space:]')"
    active_jobs="${active_jobs:-0}"
  done
}

launch_job() {
  local agent="$1"
  local seed="$2"
  local run_id="${ENV_NAME}--${agent}--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching: env=${ENV_NAME} agent=${agent} seed=${seed}"
  mkdir -p "${hydra_dir}"
  (
    if nice -n "${NICE_LEVEL}" true >/dev/null 2>&1; then
      runner=(nice -n "${NICE_LEVEL}" "${PYTHON_BIN}")
    else
      runner=("${PYTHON_BIN}")
    fi
    "${runner[@]}" train.py \
      env="${ENV_NAME}" \
      agent="${agent}" \
      seed="${seed}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      +wandb_tags="[${WANDB_TAGS}]" \
      hydra.run.dir="${hydra_dir}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "env=${ENV_NAME}"
  echo "agents=${AGENTS}"
  echo "seeds=${SEEDS}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "nice_level=${NICE_LEVEL}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "wandb_tags=${WANDB_TAGS}"
  echo "run_root=${RUN_ROOT}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
} >"${MANIFEST}"

for agent in ${AGENTS}; do
  for seed in ${SEEDS}; do
    wait_for_slot
    launch_job "${agent}" "${seed}"
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=$((failed + 1))
  fi
done

if (( failed > 0 )); then
  echo "Completed with ${failed} failed job(s). Check ${RUN_ROOT}/logs." >&2
  exit 1
fi

echo "All $((${#pids[@]})) stochasticity ablation runs complete."
echo "Run root: ${RUN_ROOT}"
echo "Next: run heldout eval for each condition using reevaluate_toy_objective_checkpoints.sh pattern."
