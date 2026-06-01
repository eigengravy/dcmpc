#!/usr/bin/env bash
set -euo pipefail

# Phase 1: Complete the λ=1e-3 slice of the DDCL ablation matrix.
# Runs all 16 remaining cells (4 Phase 0 cells already done).
#
# Dimensions at λ=1e-3:
#   CE:  7 new cells  (4 stoch × 2 prop = 8, minus 1 done)
#   SCE: 3 new cells  (2 stoch × 2 prop = 4, minus 1 done)
#   MSE: 3 new cells  (4 stoch × 1 prop = 4, minus 1 done)
#   Cos: 3 new cells  (4 stoch × 1 prop = 4, minus 1 done)
#
# Total: 16 cells × 5 seeds = 80 runs.
#
# Capacity-matched to FSQ [5,3] = 15 codes/group.
# Default config: ddcl_deltas=[0.4, 0.667], ddcl_scales=[0.8, 0.667]
#
# See: dcmpc/private/Docs/Current/ablation_experiment_plan_20260601.md

ENV_NAME="${ENV_NAME:-toy-precision-gate-final}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cpu}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/ddcl_ablation_p1_$(date +%Y%m%d_%H%M%S)}"
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

  echo "Launching: agent=${agent} variant=${variant} seed=${seed} overrides=[$*]"
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
      experiment_tag="${variant}" \
      hydra.run.dir="${hydra_dir}" \
      "$@"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

NUM_SEEDS=$(echo ${SEEDS} | wc -w | tr -d ' ')
TOTAL_JOBS=$((16 * NUM_SEEDS))

echo "=== DDCL Ablation Phase 1: Complete λ=1e-3 slice ==="
echo "  env=${ENV_NAME}"
echo "  seeds=${SEEDS} (${NUM_SEEDS} seeds)"
echo "  max_parallel=${MAX_PARALLEL}"
echo "  run_root=${RUN_ROOT}"
echo "  device=${DEVICE}"
echo "  wandb_project=${WANDB_PROJECT_NAME}"
echo "  total_jobs=${TOTAL_JOBS}"
echo ""
echo "Cells (16 new, completing 20-cell λ=1e-3 matrix):"
echo "  CE:  (d,d)+M, (s,d)+W, (s,d)+M, (d,s)+W, (d,s)+M, (s,s)+W, (s,s)+M"
echo "  SCE: (d,*)+M, (s,*)+W, (s,*)+M"
echo "  MSE: (s,d), (d,s), (s,s)"
echo "  Cos: (s,d), (d,s), (s,s)"
echo ""

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "env=${ENV_NAME}"
  echo "phase=1 (λ=1e-3 completion)"
  echo "cells=16"
  echo "seeds=${SEEDS}"
  echo "total_jobs=${TOTAL_JOBS}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "nice_level=${NICE_LEVEL}"
  echo "thread_limits=OMP:${OMP_NUM_THREADS} MKL:${MKL_NUM_THREADS} VECLIB:${VECLIB_MAXIMUM_THREADS} NUMEXPR:${NUMEXPR_NUM_THREADS}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "note=Phase 1 ablation: stoch×prop×objective at λ=1e-3, capacity-matched 15 codes"
  echo "ddcl_deltas=[0.4, 0.667] (DCMPCConfig defaults)"
  echo "ddcl_scales=[0.8, 0.667] (DCMPCConfig defaults)"
  echo "plan_doc=dcmpc/private/Docs/Current/ablation_experiment_plan_20260601.md"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

for seed in ${SEEDS}; do

  # ===================================================================
  # CE: 7 new cells (4 stoch × 2 prop = 8, minus (d,d)+W done in Phase 0)
  # ===================================================================

  # CE(d,d)+M — mode propagation, det/det
  wait_for_slot
  launch_job ddcl_ce "abl-ce-dd-M-lam3" "${seed}" \
    agent.plan_unc_prop_mode=mode

  # CE(s,d)+W — stoch eval, det targets, weighted-avg
  wait_for_slot
  launch_job ddcl_ce_stoch_eval "abl-ce-sd-W-lam3" "${seed}"

  # CE(s,d)+M — stoch eval, det targets, mode
  wait_for_slot
  launch_job ddcl_ce_stoch_eval "abl-ce-sd-M-lam3" "${seed}" \
    agent.plan_unc_prop_mode=mode

  # CE(d,s)+W — det eval, stoch targets, weighted-avg
  wait_for_slot
  launch_job ddcl_ce_stoch_tgt "abl-ce-ds-W-lam3" "${seed}"

  # CE(d,s)+M — det eval, stoch targets, mode
  wait_for_slot
  launch_job ddcl_ce_stoch_tgt "abl-ce-ds-M-lam3" "${seed}" \
    agent.plan_unc_prop_mode=mode

  # CE(s,s)+W — stoch eval, stoch targets, weighted-avg
  wait_for_slot
  launch_job ddcl_ce_stoch "abl-ce-ss-W-lam3" "${seed}"

  # CE(s,s)+M — stoch eval, stoch targets, mode
  wait_for_slot
  launch_job ddcl_ce_stoch "abl-ce-ss-M-lam3" "${seed}" \
    agent.plan_unc_prop_mode=mode

  # ===================================================================
  # SCE: 3 new cells (2 stoch × 2 prop = 4, minus (d,*)+W done in Phase 0)
  # Note: SCE target stochasticity is redundant (P2 pruning rule).
  # ===================================================================

  # SCE(d,*)+M — det eval, mode propagation
  wait_for_slot
  launch_job ddcl_soft_ce "abl-sce-d-M-lam3" "${seed}" \
    agent.plan_unc_prop_mode=mode

  # SCE(s,*)+W — stoch eval, weighted-avg
  wait_for_slot
  launch_job ddcl_soft_ce "abl-sce-s-W-lam3" "${seed}" \
    agent.ddcl_deterministic_eval=false

  # SCE(s,*)+M — stoch eval, mode propagation
  wait_for_slot
  launch_job ddcl_soft_ce "abl-sce-s-M-lam3" "${seed}" \
    agent.ddcl_deterministic_eval=false \
    agent.plan_unc_prop_mode=mode

  # ===================================================================
  # MSE: 3 new cells (4 stoch × 1 prop = 4, minus (d,d) done in Phase 0)
  # Propagation mode is N/A for MSE (continuous prediction path).
  # ===================================================================

  # MSE(s,d) — stoch eval, det targets
  wait_for_slot
  launch_job ddcl_mse "abl-mse-sd-lam3" "${seed}" \
    agent.ddcl_deterministic_eval=false

  # MSE(d,s) — det eval, stoch targets
  wait_for_slot
  launch_job ddcl_mse "abl-mse-ds-lam3" "${seed}" \
    agent.ddcl_deterministic_targets=false

  # MSE(s,s) — stoch eval, stoch targets
  wait_for_slot
  launch_job ddcl_mse "abl-mse-ss-lam3" "${seed}" \
    agent.ddcl_deterministic_eval=false \
    agent.ddcl_deterministic_targets=false

  # ===================================================================
  # Cosine: 3 new cells (4 stoch × 1 prop = 4, minus (d,d) done in Phase 0)
  # Propagation mode is N/A for Cosine (continuous prediction path).
  # ===================================================================

  # Cos(s,d) — stoch eval, det targets (uses existing config)
  wait_for_slot
  launch_job ddcl_cos_stoch_eval "abl-cos-sd-lam3" "${seed}"

  # Cos(d,s) — det eval, stoch targets
  wait_for_slot
  launch_job ddcl_cosine "abl-cos-ds-lam3" "${seed}" \
    agent.ddcl_deterministic_targets=false

  # Cos(s,s) — stoch eval, stoch targets
  wait_for_slot
  launch_job ddcl_cosine "abl-cos-ss-lam3" "${seed}" \
    agent.ddcl_deterministic_eval=false \
    agent.ddcl_deterministic_targets=false

done

echo ""
echo "All ${TOTAL_JOBS} jobs launched. Waiting for completion..."
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

echo "Phase 1 ablation sweep complete. All ${TOTAL_JOBS} jobs succeeded."
echo "Logs: ${RUN_ROOT}/logs"
echo "Next: analyze results, then run Phase 2 (λ sweep)."
