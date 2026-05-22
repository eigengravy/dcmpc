#!/usr/bin/env bash
# =============================================================================
# slurm_metaworld_singularity.sh
#
# Submits Meta-World button-press baseline experiments as a Slurm array job,
# with each task running inside the Singularity/Apptainer container.
#
# This script does NOT use Hydra Submitit. Each array element is one
# (agent, seed) combination.  This is the most portable approach because
# nested sbatch from inside an Apptainer exec is disallowed on many clusters.
#
# USAGE
# -----
#   # Build the SIF once (on a login node or machine with Docker):
#   cd dcmpc
#   docker build -t dcmpc .
#   apptainer build singularity/ddcl_mbrl.sif docker-daemon://dcmpc:latest
#
#   # Then from the repo root (dcmpc/ directory):
#   bash scripts/slurm_metaworld_singularity.sh
#
# ENVIRONMENT OVERRIDES (export before running this script)
# ---------------------------------------------------------
#   SIF_PATH         Path to the .sif image (default: singularity/ddcl_mbrl.sif)
#   AGENTS           Comma-separated agent names (default: see below)
#   SEEDS            Comma-separated seed values (default: 0,1,2,3,4)
#   ENV_NAME         Hydra env config name (default: mw-button-press)
#   WANDB_PROJECT    W&B project name (default: ddcl_mbrl_metaworld)
#   EXPERIMENT_TAG   Tag for this sweep (default: metaworld-v1-YYYYMMDD_HHMMSS)
#   PARTITION        Slurm partition (default: gpu)
#   GRES             GPU resource (default: gpu:1)
#   CPUS_PER_TASK    CPUs per task (default: 5)
#   MEM_GB           Memory in GB (default: 32)
#   TIMEOUT_H        Wall-time hours (default: 24)
#   MUJOCO_GL        OpenGL backend: egl (GPU) or osmesa (CPU-only) (default: egl)
#   OUTPUT_ROOT      Root directory for run outputs (default: output/metaworld_runs)
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/singularity/ddcl_mbrl.sif}"
AGENTS="${AGENTS:-ddcl_cosine,ddcl_ce,dcmpc,vq_ce,continuous_mse}"
SEEDS="${SEEDS:-0,1,2,3,4}"
ENV_NAME="${ENV_NAME:-mw-button-press}"
WANDB_PROJECT="${WANDB_PROJECT:-ddcl_mbrl_metaworld}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-metaworld-v1-${RUN_ID}}"
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-5}"
MEM_GB="${MEM_GB:-32}"
TIMEOUT_H="${TIMEOUT_H:-24}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/metaworld_runs}"

# ── Derived arrays ────────────────────────────────────────────────────────────
IFS=',' read -ra AGENT_ARR <<< "${AGENTS}"
IFS=',' read -ra SEED_ARR  <<< "${SEEDS}"
N_AGENTS="${#AGENT_ARR[@]}"
N_SEEDS="${#SEED_ARR[@]}"
N_JOBS=$(( N_AGENTS * N_SEEDS ))
ARRAY_END=$(( N_JOBS - 1 ))

# ── Output paths ──────────────────────────────────────────────────────────────
SWEEP_DIR="${OUTPUT_ROOT}/${ENV_NAME}_singularity_${RUN_ID}"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── Validate SIF ──────────────────────────────────────────────────────────────
if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: Singularity image not found at ${SIF_PATH}"
    echo ""
    echo "Build it first:"
    echo "  cd ${REPO_ROOT}"
    echo "  docker build -t dcmpc ."
    echo "  apptainer build singularity/ddcl_mbrl.sif docker-daemon://dcmpc:latest"
    echo ""
    echo "Or override the path:"
    echo "  SIF_PATH=/path/to/ddcl_mbrl.sif bash scripts/slurm_metaworld_singularity.sh"
    exit 1
fi

echo "============================================================"
echo "  DDCL-MBRL Meta-World Slurm Array Submission"
echo "============================================================"
echo "  SIF:           ${SIF_PATH}"
echo "  env:           ${ENV_NAME}"
echo "  agents (${N_AGENTS}):   ${AGENTS}"
echo "  seeds  (${N_SEEDS}):   ${SEEDS}"
echo "  total jobs:    ${N_JOBS}"
echo "  experiment:    ${EXPERIMENT_TAG}"
echo "  sweep_dir:     ${SWEEP_DIR}"
echo "  partition:     ${PARTITION}  gres=${GRES}"
echo "  cpus/task:     ${CPUS_PER_TASK}  mem=${MEM_GB}G  time=${TIMEOUT_H}h"
echo "  MUJOCO_GL:     ${MUJOCO_GL}"
echo "============================================================"

# ── Write the agent/seed lookup table for array tasks ─────────────────────────
LOOKUP_FILE="${SWEEP_DIR}/task_lookup.txt"
{
    echo "# task_id  agent  seed"
    task_id=0
    for agent in "${AGENT_ARR[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
            echo "${task_id} ${agent} ${seed}"
            (( task_id++ ))
        done
    done
} > "${LOOKUP_FILE}"

echo "Task lookup table written to ${LOOKUP_FILE}"

# ── Submit the array job ───────────────────────────────────────────────────────
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=ddcl_mw_${ENV_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_GB}G
#SBATCH --time=${TIMEOUT_H}:00:00
#SBATCH --array=0-${ARRAY_END}
#SBATCH --output=${LOG_DIR}/task_%a_%j.out
#SBATCH --error=${LOG_DIR}/task_%a_%j.err
#SBATCH --requeue

# ── Load cluster modules (adjust to your cluster) ─────────────────────────────
# module load apptainer   # uncomment if apptainer is a module on your cluster
# module load cuda/12.4   # uncomment if CUDA is a module on your cluster

# ── Resolve (agent, seed) from task ID ───────────────────────────────────────
TASK_LINE=\$(awk "NR==\$(( SLURM_ARRAY_TASK_ID + 2 ))" "${LOOKUP_FILE}")
TASK_ID=\$(echo "\${TASK_LINE}" | awk '{print \$1}')
AGENT=\$(echo "\${TASK_LINE}"   | awk '{print \$2}')
SEED=\$(echo "\${TASK_LINE}"    | awk '{print \$3}')

echo "============================================================"
echo "  SLURM_ARRAY_TASK_ID : \${SLURM_ARRAY_TASK_ID}"
echo "  agent               : \${AGENT}"
echo "  seed                : \${SEED}"
echo "  hostname            : \$(hostname)"
echo "  date                : \$(date)"
echo "============================================================"

# ── Per-run output directory ──────────────────────────────────────────────────
RUN_DIR="${SWEEP_DIR}/runs/\${AGENT}/seed_\${SEED}"
mkdir -p "\${RUN_DIR}"

# ── Apptainer execution ───────────────────────────────────────────────────────
# --nv            : NVIDIA GPU passthrough
# --bind repo     : bind the repo root read-write so outputs are written to host
# --bind cache    : persistent uv/mujoco cache between jobs
mkdir -p "${REPO_ROOT}/.cache"

CMD="cd /workspace && \
  export MUJOCO_GL=${MUJOCO_GL} \
         PYOPENGL_PLATFORM=${MUJOCO_GL} \
         MPLBACKEND=Agg \
         XDG_CACHE_HOME=/workspace/.cache \
         PYTHONDONTWRITEBYTECODE=1 && \
  uv run python train.py \
    env=${ENV_NAME} \
    agent=\${AGENT} \
    seed=\${SEED} \
    device=cuda \
    use_wandb=true \
    wandb_project_name=${WANDB_PROJECT} \
    experiment_tag=${EXPERIMENT_TAG} \
    num_eval_episodes=20 \
    restore_best_checkpoint_at_end=true \
    best_checkpoint_metric=episodic_success \
    best_checkpoint_tiebreaker_metric=episodic_return \
    log_best_checkpoint_eval=true \
    hydra.run.dir=\${RUN_DIR}"

apptainer exec \
    --nv \
    --bind "${REPO_ROOT}:/workspace" \
    --bind "${REPO_ROOT}/.cache:/workspace/.cache" \
    "${SIF_PATH}" \
    bash -lc "\${CMD}"

EXIT_CODE=\$?
echo "Job finished with exit code \${EXIT_CODE} at \$(date)"
exit \${EXIT_CODE}
EOF

echo ""
echo "Submitted array job with ${N_JOBS} tasks (indices 0-${ARRAY_END})."
echo "Logs: ${LOG_DIR}/task_<TASK_ID>_<JOB_ID>.{out,err}"
echo "Runs: ${SWEEP_DIR}/runs/<agent>/seed_<seed>/"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel <JOBID>"
