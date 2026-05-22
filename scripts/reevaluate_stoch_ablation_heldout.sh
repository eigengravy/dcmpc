#!/usr/bin/env bash
# Heldout evaluation for the stochasticity-ablation conditions.
#
# Re-runs the 4 stochastic conditions × 5 seeds = 20 evals that failed on
# 2026-05-19 due to absolute checkpoint paths being double-prepended by eval.py.
# This script uses relative paths (relative to dcmpc/) to avoid the bug.
#
# Usage (from dcmpc/ directory):
#   conda run -n ddcl_mbrl bash scripts/reevaluate_stoch_ablation_heldout.sh
#
# Conditions covered (the DDCL-CE (d,d) / "best-ce" condition is covered
# separately by reevaluate_toy_paper_checkpoints.sh + E1a seeds 3,4):
#   ddcl_ce_stoch      → CE stochastic eval + stochastic targets (s,s)
#   ddcl_ce_stoch_eval → CE stochastic eval + deterministic targets (s,d)
#   ddcl_ce_stoch_tgt  → CE deterministic eval + stochastic targets (d,s)
#   ddcl_cos_stoch_eval→ Cosine stochastic eval + deterministic targets (s,d)
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
EVAL_SEED="${EVAL_SEED:-10000}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy_eval}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

STOCH_ROOT="output/toy_runs/ddcl_stoch_ablation_20260519_210503/hydra"
RUN_ROOT="output/toy_runs/stoch_ablation_heldout_eval_rerun_$(date +%Y%m%d_%H%M%S)"
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
    sleep 10
    active_jobs="$(jobs -pr | wc -l | tr -d '[:space:]')"
    active_jobs="${active_jobs:-0}"
  done
}

launch_eval() {
  local label="$1"
  local checkpoint="$2"   # RELATIVE path from dcmpc/
  local safe_label="${label//\//--}"
  local log_file="${RUN_ROOT}/logs/${safe_label}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${safe_label}"
  mkdir -p "${hydra_dir}"

  echo "Launching: ${label}  (checkpoint=${checkpoint})"
  (
    "${PYTHON_BIN}" eval.py \
      checkpoint="${checkpoint}" \
      num_eval_episodes="${NUM_EVAL_EPISODES}" \
      eval_seed="${EVAL_SEED}" \
      device="${DEVICE}" \
      capture_eval_video=false \
      use_wandb="${WANDB_PROJECT_NAME:+true}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      run_name="paper-heldout-stoch_ablation--${safe_label}" \
      hydra.run.dir="${hydra_dir}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

CONDITIONS=(ddcl_ce_stoch ddcl_ce_stoch_eval ddcl_ce_stoch_tgt ddcl_cos_stoch_eval)

echo "Starting stochasticity-ablation heldout re-evaluation (20 runs)..."
echo "Output: ${RUN_ROOT}"

for cond in "${CONDITIONS[@]}"; do
  for seed in 0 1 2 3 4; do
    run_name="toy-precision-gate-final--${cond}--s${seed}"
    checkpoint="${STOCH_ROOT}/${run_name}/checkpoint.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "WARNING: checkpoint not found, skipping: ${checkpoint}" >&2
      continue
    fi
    wait_for_slot
    launch_eval "${run_name}" "${checkpoint}"
  done
done

failed_jobs=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed_jobs=$((failed_jobs + 1))
  fi
done

echo ""
echo "Run root: ${RUN_ROOT}"
if (( failed_jobs > 0 )); then
  echo "COMPLETED WITH ${failed_jobs} FAILED EVAL(S). Check logs in ${RUN_ROOT}/logs/"
  exit 1
fi

echo "All 20 evals completed. Logs: ${RUN_ROOT}/logs/"
echo "${RUN_ROOT}" > output/toy_runs/.last_stoch_ablation_heldout_root
