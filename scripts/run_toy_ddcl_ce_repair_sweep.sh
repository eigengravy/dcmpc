#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-toy-precision-gate-final}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/ddcl_ce_repair_$(date +%Y%m%d_%H%M%S)}"
NICE_LEVEL="${NICE_LEVEL:-0}"
MANIFEST="${RUN_ROOT}/manifest.txt"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs"

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
  local variant="$1"
  local seed="$2"
  shift 2
  local run_id="${ENV_NAME}--ddcl_ce--${variant}--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching DDCL-CE repair: variant=${variant} seed=${seed} log=${log_file}"
  mkdir -p "${hydra_dir}"
  (
    if [[ "${NICE_LEVEL}" == "0" ]]; then
      runner=("${PYTHON_BIN}")
    elif nice -n "${NICE_LEVEL}" true >/dev/null 2>&1; then
      runner=(nice -n "${NICE_LEVEL}" "${PYTHON_BIN}")
    else
      echo "Warning: nice level ${NICE_LEVEL} is not permitted; running without nice."
      runner=("${PYTHON_BIN}")
    fi
    "${runner[@]}" train.py \
      env="${ENV_NAME}" \
      agent=ddcl_ce \
      seed="${seed}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      experiment_tag="${variant}" \
      hydra.run.dir="${hydra_dir}" \
      "$@"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

echo "Toy DDCL-CE repair sweep"
echo "  env=${ENV_NAME}"
echo "  seeds=${SEEDS}"
echo "  max_parallel=${MAX_PARALLEL}"
echo "  run_root=${RUN_ROOT}"
echo "  wandb_project=${WANDB_PROJECT_NAME}"

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "env=${ENV_NAME}"
  echo "agent=ddcl_ce"
  echo "seeds=${SEEDS}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "nice_level=${NICE_LEVEL}"
  echo "thread_limits=OMP:${OMP_NUM_THREADS} MKL:${MKL_NUM_THREADS} VECLIB:${VECLIB_MAXIMUM_THREADS} NUMEXPR:${NUMEXPR_NUM_THREADS}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

for seed in ${SEEDS}; do
  wait_for_slot
  launch_job "ce-det-eval-wavg-targets-s35-l1e3" "${seed}" \
    agent.ddcl_deterministic_eval=true \
    agent.ddcl_deterministic_targets=true \
    agent.plan_unc_prop_mode=weighted-avg \
    agent.unc_prop_mode=sample \
    agent.ddcl_scale=3.5 \
    agent.ddcl_delta=1.0 \
    agent.ddcl_lambda=0.001

  wait_for_slot
  launch_job "ce-det-eval-mode-targets-s35-l1e3" "${seed}" \
    agent.ddcl_deterministic_eval=true \
    agent.ddcl_deterministic_targets=true \
    agent.plan_unc_prop_mode=mode \
    agent.unc_prop_mode=sample \
    agent.ddcl_scale=3.5 \
    agent.ddcl_delta=1.0 \
    agent.ddcl_lambda=0.001

  wait_for_slot
  launch_job "ce-det-eval-wavg-targets-s15-l3e4" "${seed}" \
    agent.ddcl_deterministic_eval=true \
    agent.ddcl_deterministic_targets=true \
    agent.plan_unc_prop_mode=weighted-avg \
    agent.unc_prop_mode=sample \
    agent.ddcl_scale=1.5 \
    agent.ddcl_delta=1.0 \
    agent.ddcl_lambda=0.0003

  wait_for_slot
  launch_job "ce-det-eval-wavg-targets-s20-l3e4" "${seed}" \
    agent.ddcl_deterministic_eval=true \
    agent.ddcl_deterministic_targets=true \
    agent.plan_unc_prop_mode=weighted-avg \
    agent.unc_prop_mode=sample \
    agent.ddcl_scale=2.0 \
    agent.ddcl_delta=1.0 \
    agent.ddcl_lambda=0.0003

  wait_for_slot
  launch_job "ce-det-eval-wavg-targets-s20-l1e4" "${seed}" \
    agent.ddcl_deterministic_eval=true \
    agent.ddcl_deterministic_targets=true \
    agent.plan_unc_prop_mode=weighted-avg \
    agent.unc_prop_mode=sample \
    agent.ddcl_scale=2.0 \
    agent.ddcl_delta=1.0 \
    agent.ddcl_lambda=0.0001

  wait_for_slot
  launch_job "ce-det-eval-wavg-trainwavg-targets-s20-l3e4" "${seed}" \
    agent.ddcl_deterministic_eval=true \
    agent.ddcl_deterministic_targets=true \
    agent.plan_unc_prop_mode=weighted-avg \
    agent.unc_prop_mode=weighted-avg \
    agent.ddcl_scale=2.0 \
    agent.ddcl_delta=1.0 \
    agent.ddcl_lambda=0.0003
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

echo "Completed all DDCL-CE repair jobs. Logs: ${RUN_ROOT}/logs"
