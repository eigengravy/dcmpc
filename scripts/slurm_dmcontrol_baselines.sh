#!/usr/bin/env bash
set -euo pipefail

LAUNCHER="${LAUNCHER:-slurm}"
WALKER_ENV_NAMES="${WALKER_ENV_NAMES:-walker-walk}"
LONG_ENV_NAMES="${LONG_ENV_NAMES:-dog-run,humanoid-walk,reacher-easy}"
SEEDS="${SEEDS:-0,1,2,3,4}"
AGENTS="${AGENTS:-ddcl_cosine,ddcl_ce,dcmpc,vq_ce,continuous_mse}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_dmcontrol}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-dmcontrol-paper-v1}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
RESTORE_BEST_CHECKPOINT="${RESTORE_BEST_CHECKPOINT:-true}"
BEST_CHECKPOINT_METRIC="${BEST_CHECKPOINT_METRIC:-episodic_return}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-output/dmcontrol_runs/baselines_${RUN_ID}}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1440}"
MEM_GB="${MEM_GB:-32}"
CPUS_PER_TASK="${CPUS_PER_TASK:-5}"
GRES="${GRES:-gpu:1}"

echo "Submitting DMControl baselines"
echo "  walker_env_names=${WALKER_ENV_NAMES}"
echo "  long_env_names=${LONG_ENV_NAMES}"
echo "  agents=${AGENTS}"
echo "  seeds=${SEEDS}"
echo "  sweep_root=${SWEEP_ROOT}"
echo "  experiment_tag=${EXPERIMENT_TAG}"

submit_group() {
  local label="$1"
  local env_names="$2"
  local num_episodes="$3"
  local eval_every="$4"
  if [[ -z "${env_names}" ]]; then
    return 0
  fi
  local sweep_dir="${SWEEP_ROOT}/${label}/hydra"
  echo "Submitting ${label}: env=${env_names} num_episodes=${num_episodes} eval_every=${eval_every} sweep_dir=${sweep_dir}"
  python train.py --multirun \
    "hydra/launcher=${LAUNCHER}" \
    "env=${env_names}" \
    "agent=${AGENTS}" \
    "seed=${SEEDS}" \
    "device=${DEVICE}" \
    "use_wandb=true" \
    "wandb_project_name=${WANDB_PROJECT_NAME}" \
    "experiment_tag=${EXPERIMENT_TAG}-${label}" \
    "num_episodes=${num_episodes}" \
    "eval_every_episodes=${eval_every}" \
    "num_eval_episodes=${NUM_EVAL_EPISODES}" \
    "restore_best_checkpoint_at_end=${RESTORE_BEST_CHECKPOINT}" \
    "best_checkpoint_metric=${BEST_CHECKPOINT_METRIC}" \
    "best_checkpoint_tiebreaker_metric=null" \
    "log_best_checkpoint_eval=true" \
    "hydra.sweep.dir=${sweep_dir}" \
    "hydra.sweep.subdir=\${hydra.job.num}" \
    "hydra.launcher.timeout_min=${TIMEOUT_MIN}" \
    "hydra.launcher.mem_gb=${MEM_GB}" \
    "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}" \
    "hydra.launcher.gres=${GRES}"
}

submit_group "walker-200k" "${WALKER_ENV_NAMES}" 200 10
submit_group "long-1m" "${LONG_ENV_NAMES}" 1000 50
