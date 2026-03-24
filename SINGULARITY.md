# Running DC-MPC with Singularity/Apptainer on Slurm

This repo already includes Hydra + Submitit Slurm integration in [utils/cluster_utils.py](/home/eigy/Data/Silo/rl/dcmpc/utils/cluster_utils.py), but when using Singularity/Apptainer the main task is getting the container runtime, GPU access, and MuJoCo rendering configured correctly.

This document captures a practical setup for this project.

## 1. Build the Docker image

Build the image from the repo root:

```bash
docker build -t dcmpc .
```

The current [Dockerfile](/home/eigy/Data/Silo/rl/dcmpc/Dockerfile) already includes:

- Python 3.12
- `cython==0.29.36`
- MuJoCo/OpenGL/EGL system libraries
- `ffmpeg` and FFmpeg development libraries
- `patchelf`
- `cmake`
- Python dependencies from `pyproject.toml`

The image currently defaults to:

```bash
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

That matches the runtime behavior observed for this repo better than `osmesa`.

## 2. Convert the Docker image to an Apptainer/Singularity image

If Apptainer is available:

```bash
apptainer build dcmpc.sif docker-daemon://dcmpc:latest
```

If your cluster uses `singularity` instead of `apptainer`, the command is usually equivalent:

```bash
singularity build dcmpc.sif docker-daemon://dcmpc:latest
```

Some clusters do not allow image builds on compute nodes. In that case, build the `.sif` locally or on a login node and then copy it to the cluster.

## 3. Run a quick interactive test

Before submitting a Slurm job, test that the container can start the training script:

```bash
apptainer exec --nv \
  --bind $PWD:/workspace \
  --bind $HOME/.cache:/root/.cache \
  dcmpc.sif \
  bash -lc 'cd /workspace && export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl && uv run python train.py env=walker-walk'
```

Notes:

- `--nv` is required for NVIDIA GPU passthrough.
- `--bind $PWD:/workspace` makes the repo available inside the container.
- `--bind $HOME/.cache:/root/.cache` allows cached files to persist between runs.

## 4. Submit a Slurm job

Create a script like this:

```bash
#!/bin/bash
#SBATCH --job-name=dcmpc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

module load apptainer

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

apptainer exec --nv \
  --bind $PWD:/workspace \
  --bind $HOME/.cache:/root/.cache \
  dcmpc.sif \
  bash -lc 'cd /workspace && uv run python train.py env=walker-walk'
```

Submit it with:

```bash
sbatch run_dcmpc.sbatch
```

If your system uses `singularity` instead of `apptainer`, replace the command accordingly.

## 5. Multi-run with Hydra Submitit

This repo supports Slurm multi-runs through Hydra/Submitit. The basic command is:

```bash
uv run python train.py -m env=walker-walk hydra/launcher=slurm
```

Inside a containerized Slurm environment, that only works if the cluster allows `sbatch` to be called from inside the container/job environment. Some HPC systems do, some do not.

If it does work on your cluster, use:

```bash
apptainer exec --nv \
  --bind $PWD:/workspace \
  --bind $HOME/.cache:/root/.cache \
  dcmpc.sif \
  bash -lc "cd /workspace && export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl && uv run python train.py -m env=walker-walk hydra/launcher=slurm"
```

If it does not work, submit one containerized job per `sbatch` script instead of nesting Slurm submission inside the container.

## 6. What about `~/.mujoco`?

For this repo, `~/.mujoco` is usually not required.

Reason:

- This project uses modern MuJoCo Python packages via `dm-control` and `mujoco>=3.x`.
- That is different from the older `mujoco-py` workflow that expected MuJoCo files under `~/.mujoco/mjpro*`.

In most cases, MuJoCo is handled by the Python package and its cache paths, not by a manual `~/.mujoco` install.

Still, if your cluster environment or some dependency expects it, you can bind it:

```bash
apptainer exec --nv \
  --bind $PWD:/workspace \
  --bind $HOME/.cache:/root/.cache \
  --bind $HOME/.mujoco:/root/.mujoco \
  dcmpc.sif \
  bash -lc 'cd /workspace && export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl && uv run python train.py env=walker-walk'
```

If `$HOME/.mujoco` does not exist, that is fine. It is only needed if your environment still relies on that older layout.

## 7. Cache and write-location recommendations

On HPC systems, runtime failures often come from unwritable cache directories rather than missing packages.

If needed, set:

```bash
export XDG_CACHE_HOME=/workspace/.cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

And make sure the cache directory exists on the bound filesystem.

For example:

```bash
mkdir -p .cache logs
```

Then run:

```bash
apptainer exec --nv \
  --bind $PWD:/workspace \
  dcmpc.sif \
  bash -lc 'cd /workspace && export XDG_CACHE_HOME=/workspace/.cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl && uv run python train.py env=walker-walk'
```

## 8. Common failure points

- EGL errors:
  Make sure you are using `--nv` and exporting `MUJOCO_GL=egl`.

- MuJoCo download/cache errors:
  Bind a writable cache directory such as `$HOME/.cache` or set `XDG_CACHE_HOME=/workspace/.cache`.

- Slurm nested submission issues:
  If Hydra Submitit cannot call `sbatch` from inside the container, use one `sbatch` script per run instead.

- Missing output directories:
  Create `logs/` and any output directories on the host before submission.

## 9. Minimal working command

If you want the shortest command to try first:

```bash
mkdir -p .cache logs

apptainer exec --nv \
  --bind $PWD:/workspace \
  dcmpc.sif \
  bash -lc 'cd /workspace && export XDG_CACHE_HOME=/workspace/.cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl && uv run python train.py env=walker-walk'
```
