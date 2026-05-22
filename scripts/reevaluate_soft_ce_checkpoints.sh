#!/usr/bin/env bash
set -euo pipefail

# Heldout re-evaluation for DDCL Soft CE toy checkpoints.
#
# Evaluates the best checkpoint from each seed of the Soft CE training run
# using a fixed heldout eval seed (default 10000) and 50 episodes, matching
# the protocol used for all other paper-facing methods.
#
# Usage (from dcmpc/ directory):
#   conda run -n ddcl_mbrl bash scripts/reevaluate_soft_ce_checkpoints.sh
#
# Key overrides (env vars):
#   SOURCE_ROOT   — path to the ddcl_soft_ce training run dir
#   EVAL_SEED     — RNG seed for evaluation (default: 10000, i.e. heldout)
#   MAX_PARALLEL  — concurrent eval jobs (default: 5 for local CPU)
#   USE_WANDB     — whether to log to W&B (default: true)

SOURCE_ROOT="${SOURCE_ROOT:-output/toy_runs/ddcl_soft_ce_20260518_165818}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/soft_ce_heldout_eval_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy_eval}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
EVAL_SEED="${EVAL_SEED:-10000}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
REQUIRE_ALL="${REQUIRE_ALL:-true}"
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

find_checkpoint() {
  local seed="$1"
  local checkpoint="${SOURCE_ROOT}/hydra/toy-precision-gate-final--ddcl_soft_ce--s${seed}/checkpoint.pt"
  if [[ -f "${checkpoint}" ]]; then
    printf '%s\n' "${checkpoint}"
    return 0
  fi
  return 1
}

launch_eval() {
  local seed="$1"
  local checkpoint="$2"
  local run_id="soft_ce_heldout--toy-precision-gate-final--ddcl_soft_ce--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching: ${run_id}  checkpoint=${checkpoint}"
  mkdir -p "${hydra_dir}"
  (
    "${PYTHON_BIN}" eval.py \
      checkpoint="${checkpoint}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      run_name="paper-heldout-${run_id}" \
      num_eval_episodes="${NUM_EVAL_EPISODES}" \
      eval_seed="${EVAL_SEED}" \
      capture_eval_video=false \
      hydra.run.dir="${hydra_dir}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "source_root=${SOURCE_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "num_eval_episodes=${NUM_EVAL_EPISODES}"
  echo "eval_seed=${EVAL_SEED}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "seeds=${SEEDS}"
  echo "require_all=${REQUIRE_ALL}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

missing=0
for seed in ${SEEDS}; do
  if checkpoint="$(find_checkpoint "${seed}")"; then
    wait_for_slot
    launch_eval "${seed}" "${checkpoint}"
  else
    echo "Missing checkpoint for seed=${seed} in ${SOURCE_ROOT}/hydra/toy-precision-gate-final--ddcl_soft_ce--s${seed}/" | tee -a "${MANIFEST}" >&2
    missing=$((missing + 1))
  fi
done

if (( missing > 0 )) && [[ "${REQUIRE_ALL}" == "true" ]]; then
  echo "Missing ${missing} checkpoint(s); set REQUIRE_ALL=false to evaluate available checkpoints." >&2
  exit 1
fi

failed_jobs=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed_jobs=$((failed_jobs + 1))
  fi
done

if (( failed_jobs > 0 )); then
  echo "Completed with ${failed_jobs} failed eval job(s). Check ${RUN_ROOT}/logs." >&2
  exit 1
fi

echo "All ${#pids[@]} heldout evals complete. Results in ${RUN_ROOT}/logs"
echo "Next: aggregate W&B runs tagged 'paper-heldout-soft_ce_heldout--*' from project ${WANDB_PROJECT_NAME}"
