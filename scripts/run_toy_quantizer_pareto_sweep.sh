#!/usr/bin/env bash
set -euo pipefail

# Toy Pareto sweep for quantizer families. It varies FSQ/VQ maximum codebook
# size and DDCL scale/lambda, then compares realized empirical entropy against
# performance. Use DEVICE=auto to prefer CUDA, then MPS, then CPU.

ENV_NAME="${ENV_NAME:-toy-precision-gate-final}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-auto}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy_pareto}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RUN_ROOT="${RUN_ROOT:-output/toy_runs/quantizer_pareto_$(date +%Y%m%d_%H%M%S)}"
NICE_LEVEL="${NICE_LEVEL:-0}"
SCHEMES="${SCHEMES:-fsq vq ddcl}"
DDCL_AGENT="${DDCL_AGENT:-ddcl_cosine}"
DDCL_LAMBDAS="${DDCL_LAMBDAS:-0.0001 0.0003 0.001 0.003}"
DDCL_SPECS="${DDCL_SPECS:-ddcl-s1p0-d1p0:1.0:1.0 ddcl-s2p0-d1p0:2.0:1.0 ddcl-s3p0-d1p0:3.0:1.0 ddcl-s3p5-d1p0:3.5:1.0}"
FSQ_SPECS="${FSQ_SPECS:-fsq-3x3:[3,3] fsq-4x4:[4,4] fsq-5x5:[5,5] fsq-8x8:[8,8]}"
VQ_SPECS="${VQ_SPECS:-vq-9:9 vq-16:16 vq-25:25 vq-64:64}"
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

launch_job() {
  local agent="$1"
  local variant="$2"
  local seed="$3"
  shift 3
  local run_id="${ENV_NAME}--${agent}--${variant}--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching toy quantizer Pareto: agent=${agent} variant=${variant} seed=${seed} log=${log_file}"
  mkdir -p "${hydra_dir}"
  (
    if [[ "${NICE_LEVEL}" == "0" ]]; then
      runner=("${PYTHON_BIN}")
    elif nice -n "${NICE_LEVEL}" true >/dev/null 2>&1; then
      runner=(nice -n "${NICE_LEVEL}" "${PYTHON_BIN}")
    else
      echo "Warning: nice level ${NICE_LEVEL} is not permitted; running without nice." >&2
      runner=("${PYTHON_BIN}")
    fi
    "${runner[@]}" train.py \
      env="${ENV_NAME}" \
      agent="${agent}" \
      seed="${seed}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      experiment_tag="pareto-${variant}" \
      restore_best_checkpoint_at_end=true \
      log_best_checkpoint_eval=true \
      hydra.run.dir="${hydra_dir}" \
      "$@"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "env=${ENV_NAME}"
  echo "schemes=${SCHEMES}"
  echo "seeds=${SEEDS}"
  echo "device=${DEVICE}"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "ddcl_agent=${DDCL_AGENT}"
  echo "ddcl_specs=${DDCL_SPECS}"
  echo "ddcl_lambdas=${DDCL_LAMBDAS}"
  echo "fsq_specs=${FSQ_SPECS}"
  echo "vq_specs=${VQ_SPECS}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

for seed in ${SEEDS}; do
  if [[ " ${SCHEMES} " == *" fsq "* ]]; then
    for spec in ${FSQ_SPECS}; do
      label="${spec%%:*}"
      levels="${spec#*:}"
      wait_for_slot
      launch_job dcmpc "${label}" "${seed}" \
        "agent.fsq_levels=${levels}"
    done
  fi

  if [[ " ${SCHEMES} " == *" vq "* ]]; then
    for spec in ${VQ_SPECS}; do
      label="${spec%%:*}"
      size="${spec#*:}"
      wait_for_slot
      launch_job vq_ce "${label}" "${seed}" \
        agent.vq_codebook_size="${size}" \
        agent.vq_codebook_dim=2
    done
  fi

  if [[ " ${SCHEMES} " == *" ddcl "* ]]; then
    for spec in ${DDCL_SPECS}; do
      label="${spec%%:*}"
      rest="${spec#*:}"
      scale="${rest%%:*}"
      delta="${rest#*:}"
      for lambda in ${DDCL_LAMBDAS}; do
        lambda_label="${lambda//./p}"
        lambda_label="${lambda_label//-/m}"
        wait_for_slot
        launch_job "${DDCL_AGENT}" "${label}-lambda${lambda_label}" "${seed}" \
          agent.ddcl_scale="${scale}" \
          agent.ddcl_delta="${delta}" \
          agent.ddcl_lambda="${lambda}" \
          agent.ddcl_deterministic_eval=true \
          agent.ddcl_deterministic_targets=true \
          agent.plan_unc_prop_mode=weighted-avg \
          agent.unc_prop_mode=sample
      done
    done
  fi
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

echo "Completed toy quantizer Pareto sweep. Logs: ${RUN_ROOT}/logs"
