#!/usr/bin/env bash
set -euo pipefail

# Completes the paper-facing DDCL objective toy rows by adding seeds 3 and 4
# for the selected DDCL MSE and DDCL cosine settings. This launches training.

ENV_NAME="${ENV_NAME:-toy-precision-gate-final}"
SEEDS="${SEEDS:-3 4}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/ddcl_objective_completion_$(date +%Y%m%d_%H%M%S)}"
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
  local active_jobs
  active_jobs="$(jobs -pr | wc -l | tr -d '[:space:]')"
  active_jobs="${active_jobs:-0}"
  while (( active_jobs >= MAX_PARALLEL )); do
    sleep 15
    active_jobs="$(jobs -pr | wc -l | tr -d '[:space:]')"
    active_jobs="${active_jobs:-0}"
  done
}

runner_prefix() {
  if [[ "${NICE_LEVEL}" == "0" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
  elif nice -n "${NICE_LEVEL}" true >/dev/null 2>&1; then
    printf '%s\n' "nice -n ${NICE_LEVEL} ${PYTHON_BIN}"
  else
    echo "Warning: nice level ${NICE_LEVEL} is not permitted; running without nice." >&2
    printf '%s\n' "${PYTHON_BIN}"
  fi
}

launch_job() {
  local variant="$1"
  local consistency_loss="$2"
  local seed="$3"
  local run_id="${ENV_NAME}--ddcl_mse--${variant}--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching toy DDCL objective completion: variant=${variant} seed=${seed} log=${log_file}"
  mkdir -p "${hydra_dir}"
  (
    read -r -a runner <<<"$(runner_prefix)"
    "${runner[@]}" train.py \
      env="${ENV_NAME}" \
      agent=ddcl_mse \
      seed="${seed}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      experiment_tag="${variant}" \
      agent.consistency_loss="${consistency_loss}" \
      agent.ddcl_deterministic_eval=true \
      agent.ddcl_deterministic_targets=true \
      agent.plan_unc_prop_mode=weighted-avg \
      agent.unc_prop_mode=sample \
      agent.ddcl_scale=3.5 \
      agent.ddcl_delta=1.0 \
      agent.ddcl_lambda=0.001 \
      hydra.run.dir="${hydra_dir}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "env=${ENV_NAME}"
  echo "seeds=${SEEDS}"
  echo "variants=mse-det-eval-wavg-targets-s35-l1e3 jepa-cosine-det-eval-wavg-targets-s35-l1e3"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "nice_level=${NICE_LEVEL}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

for seed in ${SEEDS}; do
  wait_for_slot
  launch_job "mse-det-eval-wavg-targets-s35-l1e3" "mse" "${seed}"

  wait_for_slot
  launch_job "jepa-cosine-det-eval-wavg-targets-s35-l1e3" "cosine" "${seed}"
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

echo "Completed DDCL objective completion jobs. Logs: ${RUN_ROOT}/logs"
