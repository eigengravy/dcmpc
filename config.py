#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Any, List, Optional

from dcmpc import DCMPCConfig
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from utils import LUMIConfig, PUHTIConfig, SlurmConfig, TritonConfig


@dataclass
class TrainConfig:
    """Training config used in train.py"""

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"agent": "dcmpc"},
            {"env": "dog-run"},  # envs are specified in cfgs/env/
            # Use submitit to launch slurm jobs on cluster w/ multirun
            {"override hydra/launcher": "slurm"},
            {"override hydra/job_logging": "colorlog"},  # Make logging colourful
            {"override hydra/hydra_logging": "colorlog"},  # Make logging colourful
        ]
    )

    # Configure environment (overridden by defaults list)
    env_name: str = MISSING
    task_name: str = MISSING
    vec_env: bool = True

    # Agent (overridden by DCMPC in defaults list)
    agent: DCMPCConfig = field(default_factory=DCMPCConfig)

    # Experiment
    max_episode_steps: int = 1000  # Max episode length
    num_episodes: int = 4000  # Number of training episodes 4000*500*2=4M steps
    random_episodes: int = 10  # Number of random episodes at start
    action_repeat: int = 2
    buffer_size: int = 10_000_000
    prefetch: int = 5
    seed: int = 42
    checkpoint: Optional[str] = None  # /file/path/to/checkpoint
    device: str = "cuda"  # "auto", "cpu", "cuda", or explicit experimental "mps"
    env_device: str = "cpu"  # DMControl/MetaWorld on cpu but maniskill/isaac on cuda
    verbose: bool = False  # if true print training progress
    log_best_checkpoint_eval: bool = True
    restore_best_checkpoint_at_end: bool = False
    best_checkpoint_metric: str = "episodic_return"
    best_checkpoint_tiebreaker_metric: Optional[str] = None
    early_stop_eval_success: Optional[float] = None
    early_stop_eval_patience: int = 1

    scale_reward: bool = False  # it true scale rewards using symlog

    # Toy Precision Gate parameters (used only when env_name="toy-precision-gate")
    toy_gate_width: float = 0.08
    toy_action_scale: float = 0.08
    toy_goal_threshold: float = 0.08
    toy_spawn_y_noise: float = 0.35
    toy_collision_terminates: bool = True
    toy_precision_success_deltas: List[float] = field(
        default_factory=lambda: [0.02, 0.04, 0.08]
    )

    # Evaluation
    eval_every_episodes: int = 25  # every 25k env steps
    num_eval_episodes: int = 10
    capture_eval_video: bool = False  # Fails on AMD GPU so set to False

    # W&B config
    use_wandb: bool = False
    wandb_silent: bool = True
    wandb_project_name: str = "DCWM"
    run_name: str = "dcmpc-${now:%Y-%m-%d_%H-%M-%S}"
    experiment_tag: Optional[str] = None

    # Override the Hydra config to get better dir structure with W&B
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "output/hydra/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"},
            "verbose": False,
            "job": {"chdir": True},
            "sweep": {"dir": "${hydra.run.dir}", "subdir": "${hydra.job.num}"},
        }
    )


@dataclass
class EvalConfig:
    """Config for eval.py - loads ckpt, creates high-res video, and logs eval metrics"""

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            # Use submitit to launch slurm jobs on cluster w/ multirun
            {"override hydra/launcher": "slurm"},
            {"override hydra/job_logging": "colorlog"},  # Make logging colourful
            {"override hydra/hydra_logging": "colorlog"},  # Make logging colourful
        ]
    )

    checkpoint: str = "output/hydra/train/2024-11-24_13-41-38/checkpoint"
    num_eval_episodes: int = 10
    eval_seed: Optional[int] = None
    capture_eval_video: bool = True
    render_size: int = 1080  # height/width of pixel observations
    device: str = "cuda"

    wandb_id: Optional[str] = None

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = "DCWM-eval"
    run_name: str = "dcmpc-eval-${now:%Y-%m-%d_%H-%M-%S}"

    # Override the Hydra config to get better dir structure with W&B
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "output/hydra/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"},
            "verbose": False,
            "job": {"chdir": True},
            "sweep": {"dir": "${hydra.run.dir}", "subdir": "${hydra.job.num}"},
        }
    )


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)
cs.store(name="dcmpc", group="agent", node=DCMPCConfig)
cs.store(name="slurm", group="hydra/launcher", node=SlurmConfig)
cs.store(name="lumi", group="hydra/launcher", node=LUMIConfig)
cs.store(name="puhti", group="hydra/launcher", node=PUHTIConfig)
cs.store(name="triton", group="hydra/launcher", node=TritonConfig)
cs.store(name="eval", node=EvalConfig)


#####################
# Experiment Configs
#####################


# --- Existing ablations (updated for quantizer field) ---


@dataclass
class ContinuousMSEConfig(DCMPCConfig):
    """Continuous latent (no quantization) + MSE consistency"""

    quantizer: str = "none"
    consistency_loss: str = "mse"


@dataclass
class DiscreteMSEConfig(DCMPCConfig):
    """FSQ discrete latent + MSE consistency"""

    consistency_loss: str = "mse"


@dataclass
class DiscreteCEDetConfig(DCMPCConfig):
    """FSQ discrete latent + CE with deterministic (inner-product) logits"""

    ce_logits_mode: str = "mse"


cs.store(name="continuous_mse", group="agent", node=ContinuousMSEConfig)
cs.store(name="discrete_mse", group="agent", node=DiscreteMSEConfig)
cs.store(name="discrete_ce_det", group="agent", node=DiscreteCEDetConfig)


# --- DDCL ablations ---


@dataclass
class DDCLCEConfig(DCMPCConfig):
    """DDCL quantizer + cross-entropy consistency"""

    quantizer: str = "ddcl"
    consistency_loss: str = "cross-entropy"
    ddcl_deterministic_eval: bool = True
    ddcl_deterministic_targets: bool = True


@dataclass
class DDCLMSEConfig(DCMPCConfig):
    """DDCL quantizer + MSE consistency"""

    quantizer: str = "ddcl"
    consistency_loss: str = "mse"
    ddcl_deterministic_eval: bool = True
    ddcl_deterministic_targets: bool = True


@dataclass
class DDCLCosineConfig(DCMPCConfig):
    """DDCL quantizer + cosine SPR/TD-MPC2-style stop-gradient latent self-prediction"""

    quantizer: str = "ddcl"
    consistency_loss: str = "cosine"
    ddcl_deterministic_eval: bool = True
    ddcl_deterministic_targets: bool = True


@dataclass
class DDCLSoftCEConfig(DCMPCConfig):
    """DDCL quantizer + analytically-derived soft CE labels from dither distribution.

    The two-bin soft label (derived from the known uniform dither distribution)
    replaces the hard CE label, eliminating label noise while giving partial credit
    for dynamics predictions in the adjacent correct bin.
    ddcl_deterministic_targets is False because soft CE computes its own analytic target.
    """

    quantizer: str = "ddcl"
    consistency_loss: str = "cross-entropy"
    ddcl_deterministic_eval: bool = True
    ddcl_deterministic_targets: bool = False  # soft CE derives targets analytically
    ddcl_soft_ce_targets: bool = True
    plan_unc_prop_mode: str = "weighted-avg"
    ddcl_scale: float = 3.5
    ddcl_delta: float = 1.0
    ddcl_lambda: float = 1e-3


cs.store(name="ddcl_ce", group="agent", node=DDCLCEConfig)
cs.store(name="ddcl_mse", group="agent", node=DDCLMSEConfig)
cs.store(name="ddcl_cosine", group="agent", node=DDCLCosineConfig)
cs.store(name="ddcl_soft_ce", group="agent", node=DDCLSoftCEConfig)


# --- DDCL stochasticity ablation configs ---
# These four variants systematically ablate det_eval and det_targets for the
# stochasticity ablation study (Table 2 in the paper).  All use scale=3.5, λ=1e-3.

@dataclass
class DDCLCEStochConfig(DCMPCConfig):
    """DDCL CE — fully stochastic: stoch eval + stoch targets.
    Baseline for the stochasticity ablation (both fixes disabled).
    """
    quantizer: str = "ddcl"
    consistency_loss: str = "cross-entropy"
    ddcl_deterministic_eval: bool = False
    ddcl_deterministic_targets: bool = False
    ddcl_scale: float = 3.5
    ddcl_lambda: float = 1e-3


@dataclass
class DDCLCEStochEvalConfig(DCMPCConfig):
    """DDCL CE — stoch eval only: stoch eval + det targets.
    Isolates the effect of planning-noise from MPPI rollouts.
    """
    quantizer: str = "ddcl"
    consistency_loss: str = "cross-entropy"
    ddcl_deterministic_eval: bool = False
    ddcl_deterministic_targets: bool = True
    ddcl_scale: float = 3.5
    ddcl_lambda: float = 1e-3


@dataclass
class DDCLCEStochTgtConfig(DCMPCConfig):
    """DDCL CE — stoch targets only: det eval + stoch targets.
    Isolates the effect of target-label stochasticity during training.
    """
    quantizer: str = "ddcl"
    consistency_loss: str = "cross-entropy"
    ddcl_deterministic_eval: bool = True
    ddcl_deterministic_targets: bool = False
    ddcl_scale: float = 3.5
    ddcl_lambda: float = 1e-3


@dataclass
class DDCLCosStochEvalConfig(DCMPCConfig):
    """DDCL Cosine — stoch eval + det targets.
    Tests whether planning-noise hurts the cosine objective (cross-check).
    """
    quantizer: str = "ddcl"
    consistency_loss: str = "cosine"
    ddcl_deterministic_eval: bool = False
    ddcl_deterministic_targets: bool = True
    plan_unc_prop_mode: str = "weighted-avg"
    ddcl_scale: float = 3.5
    ddcl_lambda: float = 1e-3


cs.store(name="ddcl_ce_stoch",      group="agent", node=DDCLCEStochConfig)
cs.store(name="ddcl_ce_stoch_eval", group="agent", node=DDCLCEStochEvalConfig)
cs.store(name="ddcl_ce_stoch_tgt",  group="agent", node=DDCLCEStochTgtConfig)
cs.store(name="ddcl_cos_stoch_eval", group="agent", node=DDCLCosStochEvalConfig)


# --- VQ ablations ---


@dataclass
class VQCEConfig(DCMPCConfig):
    """VQ quantizer + cross-entropy consistency"""

    quantizer: str = "vq"
    consistency_loss: str = "cross-entropy"


@dataclass
class VQMSEConfig(DCMPCConfig):
    """VQ quantizer + MSE consistency"""

    quantizer: str = "vq"
    consistency_loss: str = "mse"


cs.store(name="vq_ce", group="agent", node=VQCEConfig)
cs.store(name="vq_mse", group="agent", node=VQMSEConfig)


# --- FSQ level variation ablations ---


@dataclass
class FSQ8x8Config(DCMPCConfig):
    """FSQ with [8, 8] levels (64 codes)"""

    fsq_levels: List[int] = field(default_factory=lambda: [8, 8])


@dataclass
class FSQ5x5x5Config(DCMPCConfig):
    """FSQ with [5, 5, 5] levels (125 codes) -- latent_dim must be divisible by 3"""

    fsq_levels: List[int] = field(default_factory=lambda: [5, 5, 5])


cs.store(name="fsq_8x8", group="agent", node=FSQ8x8Config)
cs.store(name="fsq_5x5x5", group="agent", node=FSQ5x5x5Config)
