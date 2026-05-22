#!/usr/bin/env bash
# Complete the 5-seed DDCL-CE(d,d) set by running the two missing seeds (3, 4)
# with EXACTLY the same config as the existing seeds 0-2 in
# output/toy_runs/ddcl_ce_repair_20260513_105700 (variant ce-det-eval-wavg-targets-s35-l1e3).
# This de-confounds the objective comparison (DDCL-CE was previously only at (s,s)).
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/Caskroom/miniforge/base/envs/ddcl_mbrl/bin/python}"
ENV_NAME="toy-precision-gate-final"
SEEDS="${SEEDS:-3 4}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/ddcl_ce_dd_complete_$(date +%Y%m%d_%H%M%S)}"
VARIANT="ce-det-eval-wavg-targets-s35-l1e3"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

mkdir -p "${RUN_ROOT}/logs"
echo "DDCL-CE(d,d) completion: seeds=${SEEDS} root=${RUN_ROOT}"

pids=()
for seed in ${SEEDS}; do
  run_id="${ENV_NAME}--ddcl_ce--${VARIANT}--s${seed}"
  hydra_dir="${RUN_ROOT}/hydra/${run_id}"
  log_file="${RUN_ROOT}/logs/${run_id}.log"
  mkdir -p "${hydra_dir}"
  echo "Launching seed ${seed} -> ${log_file}"
  (
    "${PYTHON_BIN}" train.py \
      env="${ENV_NAME}" \
      agent=ddcl_ce \
      seed="${seed}" \
      device=cpu \
      use_wandb=true \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      experiment_tag="${VARIANT}" \
      hydra.run.dir="${hydra_dir}" \
      agent.ddcl_deterministic_eval=true \
      agent.ddcl_deterministic_targets=true \
      agent.plan_unc_prop_mode=weighted-avg \
      agent.unc_prop_mode=sample \
      agent.ddcl_scale=3.5 \
      agent.ddcl_delta=1.0 \
      agent.ddcl_lambda=0.001
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
done

echo "PIDs: ${pids[*]}"
echo "${pids[*]}" > "${RUN_ROOT}/pids.txt"
echo "${RUN_ROOT}" > "${RUN_ROOT}/../.last_ddcl_ce_dd_run_root"

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=$((failed+1)); fi
done
echo "Done. failed=${failed}. Logs: ${RUN_ROOT}/logs"
exit "${failed}"
