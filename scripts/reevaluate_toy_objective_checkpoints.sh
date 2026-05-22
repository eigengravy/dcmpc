#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate selected DDCL MSE/cosine toy checkpoints. SOURCE_ROOTS can contain
# one or more hydra roots, e.g. the original objective sweep plus the completion
# sweep for seeds 3 and 4.

SOURCE_ROOTS="${SOURCE_ROOTS:-output/toy_runs/ddcl_objective_sweep_20260514_073320/hydra}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/objective_checkpoint_reeval_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy_eval}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
EVAL_SEED="${EVAL_SEED:-}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SEEDS="${SEEDS:-0 1 2 3 4}"
VARIANTS="${VARIANTS:-mse-det-eval-wavg-targets-s35-l1e3 jepa-cosine-det-eval-wavg-targets-s35-l1e3}"
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
  local variant="$1"
  local seed="$2"
  local checkpoint
  for source_root in ${SOURCE_ROOTS}; do
    checkpoint="${source_root}/toy-precision-gate-final--ddcl_mse--${variant}--s${seed}/checkpoint.pt"
    if [[ -f "${checkpoint}" ]]; then
      printf '%s\n' "${checkpoint}"
      return 0
    fi
  done
  return 1
}

launch_eval() {
  local variant="$1"
  local seed="$2"
  local checkpoint="$3"
  local run_id="objective--toy-precision-gate-final--ddcl_mse--${variant}--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching objective checkpoint eval: ${run_id} checkpoint=${checkpoint}"
  mkdir -p "${hydra_dir}"
  (
    "${PYTHON_BIN}" eval.py \
      checkpoint="${checkpoint}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      run_name="paper-reeval-${run_id}" \
      num_eval_episodes="${NUM_EVAL_EPISODES}" \
      ${EVAL_SEED:+eval_seed="${EVAL_SEED}"} \
      capture_eval_video=false \
      hydra.run.dir="${hydra_dir}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "source_roots=${SOURCE_ROOTS}"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "num_eval_episodes=${NUM_EVAL_EPISODES}"
  echo "eval_seed=${EVAL_SEED:-train_seed}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "seeds=${SEEDS}"
  echo "variants=${VARIANTS}"
  echo "require_all=${REQUIRE_ALL}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

missing=0
for variant in ${VARIANTS}; do
  for seed in ${SEEDS}; do
    if checkpoint="$(find_checkpoint "${variant}" "${seed}")"; then
      wait_for_slot
      launch_eval "${variant}" "${seed}" "${checkpoint}"
    else
      echo "Missing checkpoint for variant=${variant} seed=${seed}" | tee -a "${MANIFEST}" >&2
      missing=$((missing + 1))
    fi
  done
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
  echo "Completed with ${failed_jobs} failed eval job(s). Check ${RUN_ROOT}/logs."
  exit 1
fi

echo "Completed all objective checkpoint evals. Logs: ${RUN_ROOT}/logs"
