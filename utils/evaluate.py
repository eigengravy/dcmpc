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
        eval_episodic_return = torch.mean(eval_data["next"]["episode_reward"][:, -1, 0])
        eval_episodic_return_std = torch.std(
            eval_data["next"]["episode_reward"][:, -1, 0]
        )
        success = eval_data["next"].get("success", None)
        episode_len = eval_data["next"]["step_count"][0, -1, -1]
        if success is not None:
            episodic_success = torch.mean(success.any(-1).to(torch.float))
            eval_metrics.update({"episodic_success": episodic_success})
        eval_metrics.update(summarize_rollout_info(eval_data))

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
