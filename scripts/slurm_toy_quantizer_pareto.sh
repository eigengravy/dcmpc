#!/usr/bin/env bash
set -euo pipefail

# Submit the toy quantizer Pareto sweep to a GPU cluster with Hydra/Submitit.
# Each quantizer setting is submitted separately so list-valued FSQ overrides
# are never confused with Hydra sweep syntax.

LAUNCHER="${LAUNCHER:-slurm}"
ENV_NAME="${ENV_NAME:-toy-precision-gate-final}"
SEEDS="${SEEDS:-0,1,2}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddcl_mbrl_toy_pareto}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-toy-quantizer-pareto-v1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-output/toy_runs/quantizer_pareto_${RUN_ID}}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1440}"
MEM_GB="${MEM_GB:-32}"
CPUS_PER_TASK="${CPUS_PER_TASK:-5}"
GRES="${GRES:-gpu:1}"

FSQ_SPECS="${FSQ_SPECS:-fsq-3x3:[3,3] fsq-4x4:[4,4] fsq-5x5:[5,5] fsq-8x8:[8,8]}"
VQ_SPECS="${VQ_SPECS:-vq-9:9 vq-16:16 vq-25:25 vq-64:64}"
DDCL_AGENT="${DDCL_AGENT:-ddcl_cosine}"
DDCL_SPECS="${DDCL_SPECS:-ddcl-s1p0-d1p0:1.0:1.0 ddcl-s2p0-d1p0:2.0:1.0 ddcl-s3p0-d1p0:3.0:1.0 ddcl-s3p5-d1p0:3.5:1.0}"
DDCL_LAMBDAS="${DDCL_LAMBDAS:-0.0001 0.0003 0.001 0.003}"

submit_setting() {
  local label="$1"
  local agent="$2"
  shift 2
  local sweep_dir="${SWEEP_ROOT}/${label}/hydra"
  echo "Submitting ${label}: agent=${agent} sweep_dir=${sweep_dir}"
  python train.py --multirun \
    "hydra/launcher=${LAUNCHER}" \
    "env=${ENV_NAME}" \
    "agent=${agent}" \
    "seed=${SEEDS}" \
    "device=${DEVICE}" \
    "use_wandb=true" \
    "wandb_project_name=${WANDB_PROJECT_NAME}" \
    "experiment_tag=${EXPERIMENT_TAG}-${label}" \
    "restore_best_checkpoint_at_end=true" \
    "log_best_checkpoint_eval=true" \
    "hydra.sweep.dir=${sweep_dir}" \
    "hydra.sweep.subdir=\${hydra.job.num}" \
    "hydra.launcher.timeout_min=${TIMEOUT_MIN}" \
    "hydra.launcher.mem_gb=${MEM_GB}" \
    "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}" \
    "hydra.launcher.gres=${GRES}" \
    "$@"
}

echo "Submitting toy quantizer Pareto sweep"
echo "  env=${ENV_NAME}"
echo "  seeds=${SEEDS}"
echo "  device=${DEVICE}"
echo "  sweep_root=${SWEEP_ROOT}"
echo "  wandb_project=${WANDB_PROJECT_NAME}"

for spec in ${FSQ_SPECS}; do
  label="${spec%%:*}"
  levels="${spec#*:}"
  submit_setting "${label}" dcmpc "agent.fsq_levels=${levels}"
done

for spec in ${VQ_SPECS}; do
  label="${spec%%:*}"
  size="${spec#*:}"
  submit_setting "${label}" vq_ce agent.vq_codebook_dim=2 agent.vq_codebook_size="${size}"
done

for spec in ${DDCL_SPECS}; do
  label="${spec%%:*}"
  rest="${spec#*:}"
  scale="${rest%%:*}"
  delta="${rest#*:}"
  for lambda in ${DDCL_LAMBDAS}; do
    lambda_label="${lambda//./p}"
    lambda_label="${lambda_label//-/m}"
    submit_setting "${label}-lambda${lambda_label}" "${DDCL_AGENT}" \
      agent.ddcl_scale="${scale}" \
      agent.ddcl_delta="${delta}" \
      agent.ddcl_lambda="${lambda}" \
      agent.ddcl_deterministic_eval=true \
      agent.ddcl_deterministic_targets=true \
      agent.plan_unc_prop_mode=weighted-avg \
      agent.unc_prop_mode=sample
  done
done
