# DC-MPC: Discrete Codebook Model Predictive Control
This repository is the official implementation of [DC-MPC](https://www.aidanscannell.com/dcmpc), 
presented in ["Discrete Codebook World Models for Continuous Control"](https://openreview.net/forum?id=lfRYzd8ady) at ICLR 2025.
[DC-MPC](https://www.aidanscannell.com/dcmpc) is a model-based reinforcement learning algorithm demonstrating the 
strengths of learning a discrete latent space with discrete codebook encodings.

> In reinforcement learning (RL), world models serve as internal simulators, enabling agents to predict environment dynamics and future outcomes in order to make informed decisions. While previous approaches leveraging discrete latent spaces, such as DreamerV3, have demonstrated strong performance in discrete action settings and visual control tasks, their comparative performance in state-based continuous control remains underexplored. In contrast, methods with continuous latent spaces, such as TD-MPC2, have shown notable success in state-based continuous control benchmarks. In this paper, we demonstrate that modelling discrete latent states has benefits over continuous latent states and that discrete codebook encodings are more effective representations for continuous control, compared to alternative encodings, such as one-hot and label-based encodings. Based on these insights, we introduce DCWM: **D**iscrete **C**odebook **W**orld **M**odel, a self-supervised world model with a discrete and stochastic latent space, where latent states are codes from a codebook. We combine DCWM with decision-time planning to get our model-based RL algorithm, named DC-MPC: **D**iscrete **C**odebook **M**odel **P**redictive **C**ontrol, which performs competitively against recent state-of-the-art algorithms, including TD-MPC2 and DreamerV3, on continuous control benchmarks. See our project website [www.aidanscannell.com/dcmpc](https://www.aidanscannell.com/dcmpc).


## Install instructions

### System dependencies

MuJoCo rendering requires OpenGL libraries, and video logging requires ffmpeg.

**macOS:**
```sh
brew install ffmpeg glfw glew
```

**Ubuntu/Debian:**
```sh
sudo apt install ffmpeg libglfw3-dev libglew-dev libosmesa6-dev
```

### Python dependencies

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

You might need to install PyTorch with CUDA/ROCm separately — see [pytorch.org](https://pytorch.org/get-started/locally/).

## Running experiments
Train the agent:
``` sh
python train.py env=walker-walk
```
To log metrics with W&B:
``` sh
python train.py env=walker-walk ++use_wandb=True
```
All tested tasks are listed in`cfgs/env`.

For the DDCL workshop-paper reproduction workflow, including toy scripts,
remote-cluster Meta-World/DMControl launchers, checkpoint artifact handoff, and
paper plotting, see `REPRODUCING.md`.

## Configuring experiments
This repo uses hydra for configuration.
You can easily try new hyperparameters for `DC-MPC` by overriding them on the command line. For example,
``` sh
python train.py env=walker-walk ++use_wandb=True ++agent.batch_size=1024
```
changes the batch size to be 1024 instead of the default value found in `dcmpc.py/DCMPCConfig`.

### Quantizer variants

The default quantizer is FSQ. Pre-configured agent variants are registered in `config.py` and can be selected via `agent=`:

``` sh
# DDCL quantizer + cross-entropy consistency loss
python train.py env=walker-walk agent=ddcl_ce

# DDCL quantizer + MSE consistency loss
python train.py env=walker-walk agent=ddcl_mse

# VQ quantizer + cross-entropy / MSE consistency loss
python train.py env=walker-walk agent=vq_ce
python train.py env=walker-walk agent=vq_mse

# Continuous latent (no quantization) + MSE
python train.py env=walker-walk agent=continuous_mse

# FSQ ablations
python train.py env=walker-walk agent=fsq_8x8
python train.py env=walker-walk agent=fsq_5x5x5
```

You can also override individual DDCL parameters directly:
``` sh
python train.py env=walker-walk agent=ddcl_ce agent.ddcl_deltas='[0.4,0.4]' agent.ddcl_scales='[0.8,0.8]' agent.ddcl_lambda=1e-3
```

| Parameter | Default | Description |
|---|---|---|
| `agent.quantizer` | `fsq` | Quantizer type: `fsq`, `ddcl`, `vq`, or `none` |
| `agent.ddcl_deltas` | `[0.4, 0.667]` | Per-channel bin widths (list; length = num_channels) |
| `agent.ddcl_scales` | `[0.8, 0.667]` | Per-channel tanh pre-scaling factors |
| `agent.ddcl_lambda` | `1e-3` | DDCL communication cost weight |
| `agent.consistency_loss` | `cross-entropy` | Consistency loss: `cross-entropy`, `mse`, or `cosine` |
| `agent.ddcl_deterministic_eval` | `true` | Disable dither during planning/eval |
| `agent.ddcl_deterministic_targets` | `true` | Disable dither for training targets |
| `agent.plan_unc_prop_mode` | `weighted-avg` | Planning propagation: `weighted-avg` or `mode` (CE/SCE only) |

### Slurm multi-run

You can also use hydra to submit multiple Slurm jobs directly from the command line using
``` sh
python train.py -m env=walker-walk ++use_wandb=True ++agent.batch_size=256,512 ++agent.lr=1e-4,1e-4
```
This uses `utils/cluster_utils.py/SlurmConfig` to configure the jobs, setting `timeout_min=1440` (i.e. 24hrs) and `mem_gb=32`.
If you want to run the job for longer (e.g 48hrs), you can use the following
``` sh
python train.py -m env=walker-walk ++use_wandb=True ++agent.batch_size=256,512 ++agent.lr=1e-4,1e-4 ++hydra.launcher.timeout_min=2880
```

## Project structure

```
dcmpc/
├── train.py                  # Training entrypoint (Hydra)
├── eval.py                   # Checkpoint re-evaluation
├── dcmpc.py                  # WorldModel: encoder, quantizer, dynamics, loss, planning
├── config.py                 # Agent configs (Hydra ConfigStore dataclasses)
│
├── cfgs/env/                 # Environment YAML configs
│   ├── toy-precision-gate-final.yaml
│   ├── walker-walk.yaml, reacher-hard.yaml, dog-run.yaml, ...
│   └── mw-button-press.yaml, ...
│
├── envs/                     # Environment wrappers
│   ├── toy_precision_gate.py # Precision-gate diagnostic environment
│   ├── dmcontrol.py          # DeepMind Control Suite
│   └── metaworld.py          # Meta-World
│
├── utils/
│   ├── layers.py             # DDCLQuantizer, NormedLinear, SimNorm
│   ├── buffers.py            # ReplayBuffer
│   ├── evaluate.py           # Evaluation loop
│   └── helper.py             # Misc utilities
│
├── scripts/                  # Launch, eval, metric extraction, plotting scripts
│   ├── run_toy_*.sh          # Toy experiment launchers
│   ├── run_dmcontrol_*.sh    # DMControl launchers
│   ├── run_metaworld_*.sh    # Meta-World launchers
│   ├── reevaluate_*.sh       # Checkpoint re-evaluation
│   ├── measure_*.py          # Rate metric computation
│   ├── extract_*.py          # W&B metric extraction
│   ├── gen_*_fig.py          # Paper figure generation
│   └── slurm_*.sh            # SLURM cluster launchers
│
├── tests/                    # Unit tests (42 tests)
│   ├── test_ddcl_quantizer.py
│   ├── test_soft_ce_labels.py
│   └── test_metric_invariants.py
│
├── results/
│   ├── data/                 # Published baseline CSVs (DreamerV3, SAC, TD-MPC, TD-MPC2)
│   └── plotting/             # Jupyter notebooks + paper plot scripts
│
├── docs/specs/               # Design specs
├── singularity/              # Container definition for cluster
├── REPRODUCING.md            # Full reproduction workflow
└── SINGULARITY.md            # Container setup instructions
```

# BibTeX
Please consider citing our paper:
``` bibtex
@inproceedings{scannell2025discrete,
  title     = {Discrete Codebook World Models for Continuous Control},
  author    = {Aidan Scannell and Mohammadreza Nakhaeinezhadfard and Kalle Kujanp{\"a}{\"a} and Yi Zhao and Kevin Sebastian Luck and Arno Solin and Joni Pajarinen},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=lfRYzd8ady}
}
```
