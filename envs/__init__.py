#!/usr/bin/env python3
import gymnasium as gym
from omegaconf import OmegaConf
from torchrl.envs import GymEnv, ParallelEnv, SerialEnv, StepCounter, TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    PermuteTransform,
    RenameTransform,
    Resize,
    RewardSum,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.record import VideoRecorder


_DMCONTROL = None
_METAWORLD = None
_MYOSUITE = None
_TOY_ENVS = {"toy-precision-gate"}


def _get_dmcontrol():
    global _DMCONTROL
    if _DMCONTROL is None:
        from dm_control import suite
        from envs.tasks import ball_in_cup, pendulum  # noqa: F401
        from .dmcontrol import make_env as dmcontrol_make_env

        if not getattr(suite, "_dcmpc_custom_tasks_registered", False):
            suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks("custom")
            suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
            suite._dcmpc_custom_tasks_registered = True
        _DMCONTROL = (suite, dmcontrol_make_env)
    return _DMCONTROL


def _get_metaworld():
    global _METAWORLD
    if _METAWORLD is None:
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
        from .metaworld import make_env as metaworld_make_env

        _METAWORLD = (ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE, metaworld_make_env)
    return _METAWORLD


def _get_myosuite():
    global _MYOSUITE
    if _MYOSUITE is None:
        try:
            from .myosuite import MYOSUITE_TASKS
            from .myosuite import make_env as myosuite_make_env
        except ImportError:
            MYOSUITE_TASKS = {}
            myosuite_make_env = None
        _MYOSUITE = (MYOSUITE_TASKS, myosuite_make_env)
    return _MYOSUITE


def _metaworld_env_name(env_name: str) -> str:
    return env_name.split("mw-", 1)[-1] + "-v3-goal-observable"


def _is_toy_env(env_name: str) -> bool:
    return env_name in _TOY_ENVS


def make_env(
    cfg, num_envs: int = 1, record_video: bool = False, tag: str = "eval", logger=None
):
    if not cfg.vec_env:
        env = make_env_fn(cfg, record_video=record_video)
    elif record_video:
        # assert num_envs == 1
        env = SerialEnv(
            num_envs,
            make_env_fn,
            create_env_kwargs={"cfg": cfg, "record_video": record_video},
            serial_for_single=True,
            device=cfg.device,  # puts policy on device
        )
    else:
        env = ParallelEnv(
            num_envs,
            make_env_fn,
            create_env_kwargs={"cfg": cfg},
            serial_for_single=True,
            device=cfg.device,  # puts policy on device
        )
        try:
            env.set_seed(cfg.seed)
        except AttributeError:
            env = SerialEnv(
                num_envs,
                make_env_fn,
                create_env_kwargs={"cfg": cfg},
                serial_for_single=True,
                device=cfg.device,  # puts policy on device
            )

    env.set_seed(cfg.seed)

    if record_video:
        video_rec_in_keys = "pixels"
        render_size = cfg.get("render_size", 64)
        env = env.append_transform(
            Compose(
                ToTensorImage(from_int=False, in_keys="pixels"),
                Resize(render_size, render_size),
                PermuteTransform((-2, -1, -3), in_keys=["pixels"]),
                VideoRecorder(
                    logger=logger, tag=tag, skip=4, in_keys=video_rec_in_keys
                ),
            ),
        )

    # In DMControl set cfg.agent.r_min=0 and cfg.agent.r_max=1*action_repeat
    dmcontrol = None
    if not cfg.env_name.startswith("mw-") and not _is_toy_env(cfg.env_name):
        dmcontrol = _get_dmcontrol()

    if dmcontrol and (
        (cfg.env_name, cfg.task_name) in dmcontrol[0].ALL_TASKS or cfg.env_name == "cup"
    ):
        OmegaConf.update(cfg.agent, "r_min", 0.0)
        OmegaConf.update(cfg.agent, "r_max", 1.0 * cfg.action_repeat)
    else:
        OmegaConf.update(cfg.agent, "r_min", None)
        OmegaConf.update(cfg.agent, "r_max", None)

    return env


def make_env_fn(cfg, record_video: bool = False):
    if cfg.env_name in gym.envs.registry.keys():
        env = GymEnv(
            env_name=cfg.env_name, frame_skip=cfg.action_repeat, device=cfg.env_device
        )
    elif _is_toy_env(cfg.env_name):
        from .toy_precision_gate import make_env as toy_precision_gate_make_env

        env = toy_precision_gate_make_env(
            seed=cfg.seed,
            frame_skip=cfg.action_repeat,
            device=cfg.env_device,
            max_episode_steps=cfg.max_episode_steps,
            gate_width=cfg.get("toy_gate_width", 0.08),
            action_scale=cfg.get("toy_action_scale", 0.08),
            goal_threshold=cfg.get("toy_goal_threshold", 0.08),
            spawn_y_noise=cfg.get("toy_spawn_y_noise", 0.35),
            collision_terminates=cfg.get("toy_collision_terminates", True),
        )
    elif cfg.env_name.startswith("mw-"):
        metaworld_registry, metaworld_make_env = _get_metaworld()
        env_name = _metaworld_env_name(cfg.env_name)
        if env_name not in metaworld_registry:
            raise KeyError(f"Unknown Meta-World environment: {cfg.env_name}")
        env = metaworld_make_env(
            env_name=env_name,
            from_pixels=record_video,
            seed=cfg.seed,
            frame_skip=cfg.action_repeat,
            device=cfg.env_device,
            max_episode_steps=cfg.max_episode_steps,
        )
    else:
        suite, dmcontrol_make_env = _get_dmcontrol()
        if (cfg.env_name, cfg.task_name) in suite.ALL_TASKS or cfg.env_name == "cup":
            env = dmcontrol_make_env(
                env_name=cfg.env_name,
                task_name=cfg.task_name,
                from_pixels=record_video,
                frame_skip=cfg.action_repeat,
                device=cfg.env_device,
            )
        else:
            myosuite_tasks, myosuite_make_env = _get_myosuite()
            if cfg.env_name not in myosuite_tasks:
                raise KeyError(f"Unknown environment: {cfg.env_name}")
            if myosuite_make_env is None:
                raise ImportError(
                    "MyoSuite is required for MyoSuite tasks. Install myosuite or use "
                    "the ddcl_mbrl environment for DMControl and Meta-World only."
                )
            env = myosuite_make_env(
                env_name=cfg.env_name,
                from_pixels=record_video,
                seed=cfg.seed,
                frame_skip=cfg.action_repeat,
                device=cfg.env_device,
                max_episode_steps=cfg.max_episode_steps,
            )

    env = TransformedEnv(
        env,
        Compose(
            RenameTransform(in_keys=["observation"], out_keys=["state"]),
            RenameTransform(in_keys=["state"], out_keys=[("observation", "state")]),
            DoubleToFloat(),
            StepCounter(),
            RewardSum(),
        ),
    )
    return env
