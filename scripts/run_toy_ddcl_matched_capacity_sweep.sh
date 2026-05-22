#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-toy-precision-gate-final}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DDCL_AGENTS="${DDCL_AGENTS:-ddcl_ce ddcl_mse ddcl_cosine}"
LAMBDAS="${LAMBDAS:-1e-4 5e-4 1e-3}"
DDCL_SCALE="${DDCL_SCALE:-1.0}"
DDCL_DELTA="${DDCL_DELTA:-1.0}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/ddcl_matched_capacity_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_TAG_PREFIX="${EXPERIMENT_TAG_PREFIX:-mc}"
NICE_LEVEL="${NICE_LEVEL:-10}"
MANIFEST="${RUN_ROOT}/manifest.txt"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

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

sanitize_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/}"
  echo "${value}"
}

launch_job() {
  local agent="$1"
  local lambda="$2"
  local seed="$3"
  local safe_lambda
  local safe_scale
  local safe_delta
  safe_lambda="$(sanitize_value "${lambda}")"
  safe_scale="$(sanitize_value "${DDCL_SCALE}")"
  safe_delta="$(sanitize_value "${DDCL_DELTA}")"
  local variant="${agent}-scale${safe_scale}-delta${safe_delta}-lambda${safe_lambda}"
  local wandb_tag="${EXPERIMENT_TAG_PREFIX}-${agent}-l${safe_lambda}"
  local run_id="${ENV_NAME}--${variant}--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching matched-capacity DDCL toy run: agent=${agent} scale=${DDCL_SCALE} delta=${DDCL_DELTA} lambda=${lambda} seed=${seed} log=${log_file}"
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
      agent="${agent}" \
      seed="${seed}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      experiment_tag="${wandb_tag}" \
      restore_best_checkpoint_at_end=true \
      best_checkpoint_metric=episodic_success \
      best_checkpoint_tiebreaker_metric=episodic_return \
      log_best_checkpoint_eval=true \
      agent.ddcl_deterministic_eval=true \
      agent.ddcl_deterministic_targets=true \
      agent.plan_unc_prop_mode=weighted-avg \
      agent.unc_prop_mode=sample \
      agent.ddcl_scale="${DDCL_SCALE}" \
      agent.ddcl_delta="${DDCL_DELTA}" \
      agent.ddcl_lambda="${lambda}" \
      hydra.run.dir="${hydra_dir}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

echo "Toy DDCL matched-capacity sweep"
echo "  env=${ENV_NAME}"
echo "  agents=${DDCL_AGENTS}"
echo "  seeds=${SEEDS}"
echo "  lambdas=${LAMBDAS}"
echo "  scale=${DDCL_SCALE}"
echo "  delta=${DDCL_DELTA}"
echo "  max_parallel=${MAX_PARALLEL}"
echo "  nice_level=${NICE_LEVEL}"
echo "  run_root=${RUN_ROOT}"
echo "  wandb_project=${WANDB_PROJECT_NAME}"

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "env=${ENV_NAME}"
  echo "agents=${DDCL_AGENTS}"
  echo "seeds=${SEEDS}"
  echo "lambdas=${LAMBDAS}"
  echo "ddcl_scale=${DDCL_SCALE}"
  echo "ddcl_delta=${DDCL_DELTA}"
  echo "matched_capacity_note=scale=1.0,delta=1.0,n_dims=2 gives 16 DDCL messages per group, close to FSQ/VQ 15"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "experiment_tag_prefix=${EXPERIMENT_TAG_PREFIX}"
  echo "nice_level=${NICE_LEVEL}"
  echo "thread_limits=OMP:${OMP_NUM_THREADS} MKL:${MKL_NUM_THREADS} VECLIB:${VECLIB_MAXIMUM_THREADS} NUMEXPR:${NUMEXPR_NUM_THREADS}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

for agent in ${DDCL_AGENTS}; do
  for lambda in ${LAMBDAS}; do
    for seed in ${SEEDS}; do
      wait_for_slot
      launch_job "${agent}" "${lambda}" "${seed}"
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

echo "Completed all matched-capacity DDCL toy jobs. Logs: ${RUN_ROOT}/logs"
