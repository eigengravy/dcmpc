#!/usr/bin/env bash
set -euo pipefail

# Format: env_config:num_episodes:eval_every_episodes
# With max_episode_steps=1000 and action_repeat=2, one episode is 1000 env steps.
TASK_SPECS="${TASK_SPECS:-walker-walk:200:10 dog-run:1000:50 humanoid-walk:1000:50 reacher-easy:1000:50}"
SEEDS="${SEEDS:-0 1 2 3 4}"
AGENTS="${AGENTS:-ddcl_ce dcmpc vq_ce continuous_mse}"
DEVICE="${DEVICE:-cuda}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_dmcontrol}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
RUN_ROOT="${RUN_ROOT:-output/dmcontrol_runs/baselines_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-dmcontrol-paper-v1}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
RESTORE_BEST_CHECKPOINT="${RESTORE_BEST_CHECKPOINT:-true}"
BEST_CHECKPOINT_METRIC="${BEST_CHECKPOINT_METRIC:-episodic_return}"
NICE_LEVEL="${NICE_LEVEL:-0}"
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
  while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_PARALLEL )); do
    sleep 15
  done
}

launch_job() {
  local env_name="$1"
  local num_episodes="$2"
  local eval_every="$3"
  local agent="$4"
  local seed="$5"
  local run_id="${env_name}--${agent}--s${seed}"
  local log_file="${RUN_ROOT}/logs/${run_id}.log"
  local hydra_dir="${RUN_ROOT}/hydra/${run_id}"

  echo "Launching DMControl baseline: env=${env_name} episodes=${num_episodes} eval_every=${eval_every} agent=${agent} seed=${seed} log=${log_file}"
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
      env="${env_name}" \
      agent="${agent}" \
      seed="${seed}" \
      device="${DEVICE}" \
      use_wandb="${USE_WANDB}" \
      wandb_project_name="${WANDB_PROJECT_NAME}" \
      experiment_tag="${EXPERIMENT_TAG}" \
      num_episodes="${num_episodes}" \
      eval_every_episodes="${eval_every}" \
      num_eval_episodes="${NUM_EVAL_EPISODES}" \
      restore_best_checkpoint_at_end="${RESTORE_BEST_CHECKPOINT}" \
      best_checkpoint_metric="${BEST_CHECKPOINT_METRIC}" \
      best_checkpoint_tiebreaker_metric=null \
      log_best_checkpoint_eval=true \
      hydra.run.dir="${hydra_dir}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
}

echo "DMControl baseline run"
echo "  task_specs=${TASK_SPECS}"
echo "  agents=${AGENTS}"
echo "  seeds=${SEEDS}"
echo "  max_parallel=${MAX_PARALLEL}"
echo "  run_root=${RUN_ROOT}"
echo "  wandb_project=${WANDB_PROJECT_NAME}"
echo "  best_checkpoint_metric=${BEST_CHECKPOINT_METRIC}"

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "task_specs=${TASK_SPECS}"
  echo "agents=${AGENTS}"
  echo "seeds=${SEEDS}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "run_root=${RUN_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "device=${DEVICE}"
  echo "wandb_project=${WANDB_PROJECT_NAME}"
  echo "experiment_tag=${EXPERIMENT_TAG}"
  echo "num_eval_episodes=${NUM_EVAL_EPISODES}"
  echo "restore_best_checkpoint_at_end=${RESTORE_BEST_CHECKPOINT}"
  echo "best_checkpoint_metric=${BEST_CHECKPOINT_METRIC}"
  echo "thread_limits=OMP:${OMP_NUM_THREADS} MKL:${MKL_NUM_THREADS} VECLIB:${VECLIB_MAXIMUM_THREADS} NUMEXPR:${NUMEXPR_NUM_THREADS}"
  echo "git_head=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_start"
  git status --short 2>/dev/null || true
  echo "git_status_end"
} >"${MANIFEST}"

for spec in ${TASK_SPECS}; do
  IFS=":" read -r env_name num_episodes eval_every <<<"${spec}"
  if [[ -z "${env_name}" || -z "${num_episodes}" || -z "${eval_every}" ]]; then
    echo "Invalid TASK_SPECS entry: ${spec}. Expected env:num_episodes:eval_every_episodes" >&2
    exit 2
  fi
  for agent in ${AGENTS}; do
    for seed in ${SEEDS}; do
      wait_for_slot
      launch_job "${env_name}" "${num_episodes}" "${eval_every}" "${agent}" "${seed}"
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

echo "Completed all DMControl baselines. Logs: ${RUN_ROOT}/logs"
