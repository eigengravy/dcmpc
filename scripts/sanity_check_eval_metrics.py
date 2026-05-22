#!/usr/bin/env python3
"""Preflight checks for paper evaluation metrics.

This script is intentionally lightweight: it validates the metric aggregation
helpers and the DDCL/FSQ/VQ rate accounting without launching training. Run it
before checkpoint re-evaluation sweeps.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: F401 - registers Hydra structured configs.
from dcmpc import WorldModel
from utils.evaluate import _episode_success
from utils.layers import DDCLQuantizer


def assert_close(actual: float, expected: float, name: str, tol: float = 1e-5) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def check_episode_success() -> None:
    value = torch.tensor(
        [
            [[0], [0], [0]],
            [[0], [1], [0]],
            [[1], [1], [1]],
        ]
    )
    assert_close(float(_episode_success(value)), 2.0 / 3.0, "episode success")


def check_codebook_and_entropy_metrics() -> None:
    class DummyQuantizer:
        codebook_size = 4

    wm = WorldModel.__new__(WorldModel)
    wm._quantizer = DummyQuantizer()
    wm._token_to_message = lambda flat_tokens: (None, None)

    indices = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 2],
            [1, 0, 2],
        ]
    )
    metrics = WorldModel._compute_codebook_metrics(wm, indices)
    expected_usage = (50.0 + 25.0 + 50.0) / 3.0
    assert_close(metrics["codebook/usage_percent"], expected_usage, "usage percent")
    assert_close(metrics["codebook/per_group_unique_mean"], 5.0 / 3.0, "unique mean")

    entropy_indices = torch.tensor(
        [
            [0, 0],
            [0, 0],
            [1, 0],
            [1, 0],
        ]
    )
    rate_metrics = WorldModel._compute_rate_metrics(wm, {"indices": entropy_indices})
    assert_close(
        rate_metrics["rate/empirical_entropy_bits_per_transition"],
        1.0,
        "empirical entropy",
    )
    assert_close(rate_metrics["rate/max_bits_per_transition"], 4.0, "max bits")


def check_ddcl_support_caveat() -> None:
    quantizer = DDCLQuantizer(n_dims=2, delta=1.0, scale=1.0, ddcl_lambda=1e-3)
    if quantizer.codebook_size != 16:
        raise AssertionError(f"expected nominal DDCL codebook size 16, got {quantizer.codebook_size}")

    axis = torch.linspace(-20.0, 20.0, steps=257)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    z = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    indices = quantizer(z, stochastic=False)["indices"]
    active = indices.unique().numel()
    if active != 9:
        raise AssertionError(
            "expected deterministic scale=1, delta=1 DDCL to reach 9 grouped "
            f"messages on a wide input sweep, got {active}"
        )


def check_hydra_configs() -> None:
    cfg_dir = str(Path(__file__).resolve().parents[1] / "cfgs")
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
            cfg = compose(
                config_name="train",
                overrides=[
                    "env=toy-precision-gate-final",
                    "agent=ddcl_cosine",
                    "use_wandb=false",
                ],
            )
    finally:
        GlobalHydra.instance().clear()

    if cfg.best_checkpoint_metric != "episodic_success":
        raise AssertionError("toy final config must checkpoint by episodic_success")
    if cfg.best_checkpoint_tiebreaker_metric != "episodic_return":
        raise AssertionError("toy final config must use episodic_return tiebreaker")
    if not cfg.agent.ddcl_deterministic_eval:
        raise AssertionError("DDCL paper configs should default to deterministic eval")
    if not cfg.agent.ddcl_deterministic_targets:
        raise AssertionError("DDCL paper configs should default to deterministic targets")


def main() -> int:
    check_episode_success()
    check_codebook_and_entropy_metrics()
    check_ddcl_support_caveat()
    check_hydra_configs()
    print("sanity_check_eval_metrics: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
