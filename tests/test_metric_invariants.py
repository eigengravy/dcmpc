import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import config  # noqa: F401 - registers Hydra structured agent configs.
from dcmpc import WorldModel
from utils.evaluate import _episode_success
from utils.layers import DDCLQuantizer


class MetricInvariantTests(unittest.TestCase):
    def test_episode_success_reduces_over_time_then_episodes(self):
        cases = [
            (torch.tensor([[0], [0], [1], [0]]), 1.0),
            (torch.tensor([0, 0, 0, 0]), 0.0),
            (
                torch.tensor(
                    [
                        [[0], [0], [0]],
                        [[0], [1], [0]],
                        [[1], [1], [1]],
                    ]
                ),
                2.0 / 3.0,
            ),
            (torch.tensor([[0, 0, 0], [0, 1, 0], [1, 1, 1]]), 2.0 / 3.0),
        ]

        for value, expected in cases:
            with self.subTest(shape=tuple(value.shape)):
                self.assertAlmostEqual(float(_episode_success(value)), expected)

    def test_codebook_usage_is_averaged_per_group(self):
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
        self.assertAlmostEqual(metrics["codebook/usage_percent"], expected_usage, places=5)
        self.assertAlmostEqual(metrics["active_percent"], expected_usage, places=5)
        self.assertAlmostEqual(metrics["codebook/per_group_unique_mean"], 5.0 / 3.0, places=5)
        self.assertEqual(metrics["codebook/per_group_usage_min"], 25.0)
        self.assertEqual(metrics["codebook/per_group_usage_max"], 50.0)

    def test_empirical_entropy_is_summed_across_groups(self):
        wm = WorldModel.__new__(WorldModel)
        indices = torch.tensor(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]
        )

        entropy = WorldModel._empirical_entropy_bits(wm, indices, codebook_size=4)
        self.assertAlmostEqual(entropy, 2.0)

    def test_rate_metrics_use_entropy_for_realized_rate(self):
        class DummyQuantizer:
            codebook_size = 4

        wm = WorldModel.__new__(WorldModel)
        wm._quantizer = DummyQuantizer()
        wm._token_to_message = lambda flat_tokens: (None, None)

        indices = torch.tensor(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
            ]
        )
        metrics = WorldModel._compute_rate_metrics(wm, {"indices": indices})

        self.assertEqual(metrics["rate/max_bits_per_transition"], 4.0)
        self.assertEqual(metrics["rate/allocated_bits_per_transition"], 4.0)
        self.assertAlmostEqual(metrics["rate/empirical_entropy_bits_per_transition"], 1.0)
        self.assertAlmostEqual(metrics["rate/empirical_entropy_ratio_vs_max"], 0.25)

    def test_ddcl_deterministic_quantization_is_repeatable(self):
        quantizer = DDCLQuantizer(
            deltas=[0.4, 0.4],
            scales=[0.8, 0.8],
            ddcl_lambda=1e-3,
        )
        z = torch.tensor([[0.2, -0.7, 1.2, -1.8]], dtype=torch.float32)

        first = quantizer(z, stochastic=False)["codes"]
        second = quantizer(z, stochastic=False)["codes"]

        self.assertTrue(torch.equal(first, second))

    def test_ddcl_cosine_config_composes(self):
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

        self.assertEqual(cfg.agent.quantizer, "ddcl")
        self.assertEqual(cfg.agent.consistency_loss, "cosine")
        self.assertTrue(cfg.agent.ddcl_deterministic_eval)
        self.assertTrue(cfg.agent.ddcl_deterministic_targets)


if __name__ == "__main__":
    unittest.main()
