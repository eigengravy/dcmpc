#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mw-button-press}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LAMBDAS="${LAMBDAS:-0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3}"
DELTAS="${DELTAS:-0.5 1.0 2.0}"
DEVICE="${DEVICE:-cuda}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_metaworld}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
RUN_ROOT="${RUN_ROOT:-output/metaworld_runs/${ENV_NAME}_ddcl_sensitivity_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-metaworld-ddcl-sensitivity-v1}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}"
RESTORE_BEST_CHECKPOINT="${RESTORE_BEST_CHECKPOINT:-true}"
NICE_LEVEL="${NICE_LEVEL:-0}"
MANIFEST="${RUN_ROOT}/manifest.txt"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

pids=()

wait_for_slot() {
  while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_PARALLEL )); do
    sleep 15
  done
}

sanitize_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/}"
  echo "${value}"
}

launch_job() {
  local lambda="$1"
  local delta="$2"
  local seed="$3"
  local safe_lambda
  local safe_delta
  safe_lambda="$(sanitize_value "${lambda}")"
  safe_delta="$(sanitize_value "${delta}")"
  local run_id="${ENV_NAME}--ddcl_ce--lambda${safe_lambda}--delta${safe_delta}--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching Meta-World DDCL sensitivity: env=${ENV_NAME} lambda=${lambda} delta=${delta} seed=${seed} project=${WANDB_PROJECT_NAME} log=${log_file}"
  mkdir -p "${hydra_dir}"
  (
    nice -n "${NICE_LEVEL}" "${PYTHON_BIN}" train.py \
      env="${ENV_NAME}" \
      agent=ddcl_ce \
      seed="${seed}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      experiment_tag="${EXPERIMENT_TAG}-lambda${safe_lambda}-delta${safe_delta}" \
      num_eval_episodes="${NUM_EVAL_EPISODES}" \
      restore_best_checkpoint_at_end="${RESTORE_BEST_CHECKPOINT}" \
      log_best_checkpoint_eval=true \
      agent.ddcl_lambda="${lambda}" \
      agent.ddcl_delta="${delta}" \
      hydra.run.dir="${hydra_dir}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

echo "Meta-World DDCL sensitivity run"
echo "  env=${ENV_NAME}"
echo "  lambdas=${LAMBDAS}"
echo "  deltas=${DELTAS}"
echo "  seeds=${SEEDS}"
echo "  max_parallel=${MAX_PARALLEL}"
echo "  run_root=${RUN_ROOT}"
echo "  experiment_tag=${EXPERIMENT_TAG}"

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "env=${ENV_NAME}"
  echo "lambdas=${LAMBDAS}"
  echo "deltas=${DELTAS}"
  echo "seeds=${SEEDS}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "experiment_tag=${EXPERIMENT_TAG}"
  echo "num_eval_episodes=${NUM_EVAL_EPISODES}"
  echo "restore_best_checkpoint_at_end=${RESTORE_BEST_CHECKPOINT}"
  echo "thread_limits=OMP:${OMP_NUM_THREADS} MKL:${MKL_NUM_THREADS} VECLIB:${VECLIB_MAXIMUM_THREADS} NUMEXPR:${NUMEXPR_NUM_THREADS}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

for lambda in ${LAMBDAS}; do
  for delta in ${DELTAS}; do
    for seed in ${SEEDS}; do
      wait_for_slot
      launch_job "${lambda}" "${delta}" "${seed}"
    done
  done
done

failed_jobs=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed_jobs=$((failed_jobs + 1))
  fi
done

if (( failed_jobs > 0 )); then
  echo "Completed with ${failed_jobs} failed job(s). Check ${RUN_ROOT}/logs."
  exit 1
fi

echo "Completed all Meta-World DDCL sensitivity jobs. Logs: ${RUN_ROOT}/logs"
