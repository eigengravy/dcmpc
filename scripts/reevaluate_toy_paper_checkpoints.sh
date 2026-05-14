#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy_eval}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/paper_checkpoint_reeval_$(date +%Y%m%d_%H%M%S)}"
FINAL_ROOT="${FINAL_ROOT:-output/toy_runs/final_v1_20260512_234000/hydra}"
CE_ROOT="${CE_ROOT:-output/toy_runs/ddcl_ce_repair_20260513_105700/hydra}"
CE_VARIANT="${CE_VARIANT:-ce-det-eval-wavg-targets-s35-l1e3}"
MANIFEST="${RUN_ROOT}/manifest.txt"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/hydra"

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

launch_eval() {
  local label="$1"
  local checkpoint="$2"
  local safe_label="${label//\//--}"
  local log_file="${RUN_ROOT}/logs/${safe_label}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${safe_label}"

  echo "Launching checkpoint re-eval: ${label} checkpoint=${checkpoint}"
  mkdir -p "${hydra_dir}"
  (
    "${PYTHON_BIN}" eval.py \
      checkpoint="${checkpoint}" \
      num_eval_episodes="${NUM_EVAL_EPISODES}" \
      capture_eval_video=false \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      run_name="paper-reeval-${safe_label}" \
      hydra.run.dir="${hydra_dir}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "run_root=${RUN_ROOT}"
  echo "final_root=${FINAL_ROOT}"
  echo "ce_root=${CE_ROOT}"
  echo "ce_variant=${CE_VARIANT}"
  echo "num_eval_episodes=${NUM_EVAL_EPISODES}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

for checkpoint in "${FINAL_ROOT}"/*/checkpoint.pt; do
  [[ -e "${checkpoint}" ]] || continue
  run_name="$(basename "$(dirname "${checkpoint}")")"
  wait_for_slot
  launch_eval "final-v1--${run_name}" "${checkpoint}"
done

for checkpoint in "${CE_ROOT}"/*"${CE_VARIANT}"*/checkpoint.pt; do
  [[ -e "${checkpoint}" ]] || continue
  run_name="$(basename "$(dirname "${checkpoint}")")"
  wait_for_slot
  launch_eval "best-ce--${run_name}" "${checkpoint}"
done

failed_jobs=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed_jobs=$((failed_jobs + 1))
  fi
done

if (( failed_jobs > 0 )); then
  echo "Completed with ${failed_jobs} failed re-eval job(s). Check ${RUN_ROOT}/logs."
  exit 1
fi

echo "Completed all checkpoint re-evals. Logs: ${RUN_ROOT}/logs"
