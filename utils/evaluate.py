#!/usr/bin/env python3
import time

import torch
from tensordict.nn import TensorDictModule


INFO_METRIC_KEYS = (
    "success",
    "near_object",
    "grasp_success",
    "grasp_reward",
    "in_place_reward",
    "obj_to_target",
    "unscaled_reward",
    "inside_gate",
    "gate_region",
    "collision",
    "goal_reached",
    "distance_to_goal",
    "distance_to_gate",
    "crossed_gate",
    "y_error_at_gate",
    "progress",
    "distance_to_button",
    "distance_to_handle",
    "door_angle",
)


def summarize_rollout_info(data, keys=INFO_METRIC_KEYS):
    """Aggregate optional environment info signals from a rollout TensorDict."""
    metrics = {}
    next_data = data.get("next", None)
    if next_data is None:
        return metrics

    for key in keys:
        value = next_data.get(key, None)
        if value is None or not torch.is_tensor(value):
            continue
        is_numeric = (
            torch.is_floating_point(value)
            or value.dtype == torch.bool
            or value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
            or value.dtype in (torch.uint8,)
        )
        if value.numel() == 0 or not is_numeric:
            continue

        value = value.to(torch.float)
        safe_key = key.replace("/", "_")
        metrics[f"{safe_key}_mean"] = value.mean()
        metrics[f"{safe_key}_max"] = value.amax()
        if value.ndim >= 3:
            final_value = value[:, -1].mean()
        elif value.ndim >= 2 and value.shape[-1] == 1:
            final_value = value[-1].mean()
        elif value.ndim >= 2:
            final_value = value[:, -1].mean()
        else:
            final_value = value[-1]
        metrics[f"{safe_key}_final"] = final_value

    return metrics


def summarize_continuous_control(data):
    """Aggregate generic diagnostics for continuous-control environments."""
    metrics = {}
    next_data = data.get("next", None)
    if next_data is None:
        return metrics

    observation = data.get("observation", None)
    next_observation = next_data.get("observation", None)
    if observation is not None and next_observation is not None:
        state = observation.get("state", None)
        next_state = next_observation.get("state", None)
        if torch.is_tensor(state) and torch.is_tensor(next_state):
            state_norm = next_state.to(torch.float).norm(dim=-1)
            metrics["control/state_norm_mean"] = state_norm.mean()
            metrics["control/state_norm_max"] = state_norm.amax()
            metrics["control/state_norm_final"] = _last_time_mean(state_norm)
            if state.shape == next_state.shape:
                state_delta_norm = (next_state - state).to(torch.float).norm(dim=-1)
                metrics["control/state_delta_norm_mean"] = state_delta_norm.mean()
                metrics["control/state_delta_norm_max"] = state_delta_norm.amax()
                metrics["control/state_delta_norm_final"] = _last_time_mean(
                    state_delta_norm
                )

    action = data.get("action", None)
    if torch.is_tensor(action):
        action = action.to(torch.float)
        action_norm = action.norm(dim=-1)
        metrics["control/action_norm_mean"] = action_norm.mean()
        metrics["control/action_norm_max"] = action_norm.amax()
        metrics["control/action_abs_mean"] = action.abs().mean()
        metrics["control/action_saturation_frac"] = (
            action.abs() > 0.95
        ).to(torch.float).mean()

    reward = next_data.get("reward", None)
    if torch.is_tensor(reward):
        reward = reward.to(torch.float)
        metrics["control/reward_step_mean"] = reward.mean()
        metrics["control/reward_step_std"] = (
            reward.std() if reward.numel() > 1 else torch.zeros((), device=reward.device)
        )
        metrics["control/reward_step_final"] = _last_time_mean(reward)

    return metrics


def _last_time_mean(value):
    if value.ndim >= 2:
        return value[:, -1].mean()
    if value.ndim == 1:
        return value[-1]
    return value.mean()


def _final_episode_value(value):
    if value.ndim >= 3:
        return value[:, -1, 0]
    if value.ndim >= 2:
        return value[-1, 0]
    return value[-1]


def _episode_success(success):
    if success.ndim >= 2:
        return success.any(dim=-1).to(torch.float).mean()
    return success.any().to(torch.float)


def evaluate(
    env,
    eval_policy_module: TensorDictModule,
    max_episode_steps: int,
    action_repeat: int = 2,
    video_env=None,
    return_rollout: bool = False,
):
    """Calculate avg. episodic return (optionally avg. success)"""
    eval_metrics = {}
    with torch.no_grad():
        eval_start_time = time.time()
        eval_data = env.rollout(
            max_steps=max_episode_steps // action_repeat,
            policy=eval_policy_module,
        )
        eval_episode_time = time.time() - eval_start_time
        episode_returns = _final_episode_value(eval_data["next"]["episode_reward"])
        eval_episodic_return = episode_returns.mean()
        eval_episodic_return_std = (
            episode_returns.std() if episode_returns.numel() > 1 else torch.zeros(())
        )
        success = eval_data["next"].get("success", None)
        episode_len = _final_episode_value(eval_data["next"]["step_count"])
        if success is not None:
            episodic_success = _episode_success(success)
            eval_metrics.update({"episodic_success": episodic_success})
        eval_metrics.update(summarize_rollout_info(eval_data))
        eval_metrics.update(summarize_continuous_control(eval_data))

    ##### Eval metrics #####
    eval_metrics.update(
        {
            "episodic_return": eval_episodic_return,
            "episodic_return_std": eval_episodic_return_std,
            "episode_time": eval_episode_time,
            "episode_len": episode_len,
            "action_repeat": action_repeat,
            "max_episode_steps": max_episode_steps,
        }
    )

    if video_env is not None:
        with torch.no_grad():
            video_env.rollout(
                max_steps=max_episode_steps // action_repeat,
                policy=eval_policy_module,
            )
        video_env.transform.dump()

    if return_rollout:
        return eval_metrics, eval_data
    return eval_metrics
