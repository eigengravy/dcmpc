#!/usr/bin/env bash
set -euo pipefail

# Run all DDCL variants on the toy problem with the new per-channel quantizer.
# Capacity-matched to FSQ [5,3] = 15 codes/group.
# Default config: ddcl_deltas=[0.4, 0.667], ddcl_scales=[0.8, 0.667]

ENV_NAME="${ENV_NAME:-toy-precision-gate-final}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cpu}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/ddcl_perchannel_$(date +%Y%m%d_%H%M%S)}"
NICE_LEVEL="${NICE_LEVEL:-10}"
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
  local variant="$2"
  local seed="$3"
  shift 3
  local run_id="${ENV_NAME}--${agent}--${variant}--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching: agent=${agent} variant=${variant} seed=${seed}"
  mkdir -p "${hydra_dir}"
  (
    if [[ "${NICE_LEVEL}" == "0" ]]; then
      runner=("${PYTHON_BIN}")
    elif nice -n "${NICE_LEVEL}" true >/dev/null 2>&1; then
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
      experiment_tag="perchannel-${variant}" \
      hydra.run.dir="${hydra_dir}" \
      "$@"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

echo "=== Toy DDCL per-channel sweep ==="
echo "  env=${ENV_NAME}"
echo "  seeds=${SEEDS}"
echo "  max_parallel=${MAX_PARALLEL}"
echo "  run_root=${RUN_ROOT}"
echo "  device=${DEVICE}"
echo "  wandb_project=${WANDB_PROJECT_NAME}"
echo ""
echo "DDCL variants (all capacity-matched to FSQ [5,3] = 15 codes/group):"
echo "  1. ddcl_ce       — CE consistency, det eval, det targets"
echo "  2. ddcl_mse      — MSE consistency, det eval, det targets"
echo "  3. ddcl_cosine   — Cosine consistency, det eval, det targets"
echo "  4. ddcl_soft_ce  — Soft CE consistency, det eval, analytic targets"
echo ""

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "env=${ENV_NAME}"
  echo "agents=ddcl_ce ddcl_mse ddcl_cosine ddcl_soft_ce"
  echo "seeds=${SEEDS}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "nice_level=${NICE_LEVEL}"
  echo "thread_limits=OMP:${OMP_NUM_THREADS} MKL:${MKL_NUM_THREADS} VECLIB:${VECLIB_MAXIMUM_THREADS} NUMEXPR:${NUMEXPR_NUM_THREADS}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "note=per-channel DDCL quantizer, capacity-matched to FSQ [5,3]=15 codes"
  echo "ddcl_deltas=[0.4, 0.667] (from DCMPCConfig defaults)"
  echo "ddcl_scales=[0.8, 0.667] (from DCMPCConfig defaults)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

for seed in ${SEEDS}; do
  # 1. DDCL-CE (d,d) — deterministic eval + deterministic targets
  wait_for_slot
  launch_job ddcl_ce "ce-dd" "${seed}"

  # 2. DDCL-MSE (d,d) — deterministic eval + deterministic targets
  wait_for_slot
  launch_job ddcl_mse "mse-dd" "${seed}"

  # 3. DDCL-Cosine (d,d) — deterministic eval + deterministic targets
  wait_for_slot
  launch_job ddcl_cosine "cos-dd" "${seed}"

  # 4. DDCL-SoftCE — deterministic eval + analytic soft targets
  wait_for_slot
  launch_job ddcl_soft_ce "sce-dd" "${seed}"
done

echo ""
echo "All jobs launched. Total: $((4 * $(echo ${SEEDS} | wc -w | tr -d ' '))) jobs."
echo "Logs: ${RUN_ROOT}/logs/"
echo ""

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

echo "All DDCL per-channel toy jobs completed. Logs: ${RUN_ROOT}/logs"
