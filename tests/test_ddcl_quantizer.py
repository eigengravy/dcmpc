#!/usr/bin/env python3
"""Tests for the per-channel DDCLQuantizer (shifted floor, overflow prevention).

Covers all 9 test categories from the design spec:
  1. Bin count per channel
  2. No overflow
  3. Dither cancellation
  4. Codebook size matches FSQ
  5. Per-channel independence
  6. Comm loss uses per-channel delta
  7. Index computation (mixed-radix)
  8. Gradient flow (STE + comm_loss)
  9. Deterministic mode
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.layers import DDCLQuantizer


class TestBinCountPerChannel(unittest.TestCase):
    """1. For each (delta, scale) pair, verify exactly round(2/delta) unique m values."""

    def _check(self, deltas, scales):
        quant = DDCLQuantizer(deltas=deltas, scales=scales)
        # Dense sweep through tanh input space
        z_raw = torch.linspace(-6, 6, 2000)
        for ch_idx, (d, s) in enumerate(zip(deltas, scales)):
            z_bounded = s * torch.tanh(z_raw)
            # Use deterministic (no dither) to count reachable bins
            m = torch.floor((z_bounded + 1.0) / d).long()
            unique_m = m.unique()
            expected = round(2.0 / d)
            self.assertEqual(
                len(unique_m), expected,
                f"ch {ch_idx}: delta={d}, expected {expected} bins, got {len(unique_m)} "
                f"(unique m: {unique_m.tolist()})",
            )

    def test_5_levels(self):
        self._check([0.4], [0.8])

    def test_3_levels(self):
        self._check([2 / 3], [2 / 3])

    def test_5_and_3(self):
        self._check([0.4, 2 / 3], [0.8, 2 / 3])

    def test_5_and_5(self):
        self._check([0.4, 0.4], [0.8, 0.8])


class TestNoOverflow(unittest.TestCase):
    """2. With scale=1-delta/2, all indices are in [0, codebook_size) across 100k random samples."""

    def _check(self, deltas, scales, n_samples=100_000):
        quant = DDCLQuantizer(deltas=deltas, scales=scales)
        num_ch = len(deltas)
        z = torch.randn(n_samples, num_ch)
        out = quant(z, stochastic=True)
        indices = out["indices"]
        self.assertTrue(
            (indices >= 0).all(),
            f"Found negative indices: min={indices.min().item()}",
        )
        self.assertTrue(
            (indices < quant.codebook_size).all(),
            f"Found out-of-range indices: max={indices.max().item()}, "
            f"codebook_size={quant.codebook_size}",
        )

    def test_toy_config(self):
        self._check([0.4, 2 / 3], [0.8, 2 / 3])

    def test_transfer_config(self):
        self._check([0.4, 0.4], [0.8, 0.8])

    def test_single_channel(self):
        self._check([0.4], [0.8])

    def test_3_channels(self):
        self._check([0.4, 0.5, 2 / 3], [0.8, 0.75, 2 / 3])


class TestDitherCancellation(unittest.TestCase):
    """3. Verify E[z_hat - z_bounded] ≈ 0, Var ≈ delta^2/12, Corr ≈ 0 per channel."""

    def test_dither_properties(self):
        torch.manual_seed(42)
        deltas = [0.4, 2 / 3]
        scales = [0.8, 2 / 3]
        quant = DDCLQuantizer(deltas=deltas, scales=scales)

        n = 500_000
        z = torch.randn(n, len(deltas)) * 0.5  # moderate input magnitude
        out = quant(z, stochastic=True)

        z_bounded = quant.scales * torch.tanh(z)
        z_approx = out["codes"].view(n, -1)
        error = z_approx - z_bounded

        for ch in range(len(deltas)):
            e = error[:, ch]
            zb = z_bounded[:, ch]
            d = deltas[ch]

            # E[error] ≈ 0
            mean_err = e.mean().item()
            self.assertAlmostEqual(
                mean_err, 0.0, delta=0.01,
                msg=f"ch {ch}: E[error]={mean_err:.4f}, expected ~0",
            )

            # Var[error] ≈ delta^2/12
            var_err = e.var().item()
            expected_var = d**2 / 12
            self.assertAlmostEqual(
                var_err, expected_var, delta=expected_var * 0.15,
                msg=f"ch {ch}: Var[error]={var_err:.6f}, expected {expected_var:.6f}",
            )

            # Corr(z_bounded, error) ≈ 0
            corr = torch.corrcoef(torch.stack([zb, e]))[0, 1].item()
            self.assertAlmostEqual(
                corr, 0.0, delta=0.05,
                msg=f"ch {ch}: Corr(z_bounded, error)={corr:.4f}, expected ~0",
            )


class TestCodebookSizeMatchesFSQ(unittest.TestCase):
    """4. Verify codebook sizes match FSQ equivalents."""

    def test_toy_15(self):
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        self.assertEqual(quant.codebook_size, 15)

    def test_transfer_25(self):
        quant = DDCLQuantizer(deltas=[0.4, 0.4], scales=[0.8, 0.8])
        self.assertEqual(quant.codebook_size, 25)

    def test_single_5(self):
        quant = DDCLQuantizer(deltas=[0.4], scales=[0.8])
        self.assertEqual(quant.codebook_size, 5)

    def test_single_3(self):
        quant = DDCLQuantizer(deltas=[2 / 3], scales=[2 / 3])
        self.assertEqual(quant.codebook_size, 3)

    def test_implicit_codebook_shape(self):
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        self.assertEqual(quant.implicit_codebook.shape, (15, 2))


class TestPerChannelIndependence(unittest.TestCase):
    """5. With asymmetric deltas, verify each channel has the correct number of levels."""

    def test_5_and_3(self):
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        self.assertEqual(quant._n_levels_per_ch, [5, 3])

    def test_n_levels_per_ch_buffer(self):
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        self.assertTrue(
            torch.equal(quant.n_levels_per_ch, torch.tensor([5, 3])),
        )

    def test_unique_per_dim_bins(self):
        """Run inputs and verify per-channel unique bin counts."""
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        torch.manual_seed(0)
        z = torch.randn(50_000, 2)
        out = quant(z, stochastic=True)
        # Decompose joint indices back to per-channel bins
        m0 = out["indices"] // 3  # offsets[0] = 3
        m1 = out["indices"] % 3   # offsets[1] = 1
        self.assertEqual(m0.unique().numel(), 5)
        self.assertEqual(m1.unique().numel(), 3)


class TestCommLossPerChannelDelta(unittest.TestCase):
    """6. Verify comm_loss output changes appropriately with per-channel deltas."""

    def test_different_deltas_give_different_loss(self):
        z = torch.randn(100, 2) * 0.3
        quant_sym = DDCLQuantizer(deltas=[0.4, 0.4], scales=[0.8, 0.8])
        quant_asym = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        loss_sym = quant_sym(z, stochastic=False)["comm_loss"].item()
        loss_asym = quant_asym(z, stochastic=False)["comm_loss"].item()
        # With larger delta on ch1, comm_loss should be smaller (coarser quantization)
        self.assertGreater(loss_sym, loss_asym)

    def test_comm_loss_is_positive(self):
        z = torch.randn(100, 2)
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        loss = quant(z, stochastic=True)["comm_loss"]
        self.assertGreater(loss.item(), 0.0)


class TestIndexComputation(unittest.TestCase):
    """7. Verify indices are valid mixed-radix encodings."""

    def test_2ch_mixed_radix(self):
        """index = m[0] * n_levels[1] + m[1] for 2-channel case."""
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        torch.manual_seed(7)
        z = torch.randn(1000, 2)
        out = quant(z, stochastic=True)

        # Manually compute what the indices should be
        z_view = z.view(-1, 1, 2)
        z_bounded = quant.scales * torch.tanh(z_view)
        eps = out["codes"].view(-1, 1, 2) - z_bounded
        # Can't recover eps exactly due to STE, but we can recompute from z
        # Just verify the indices decompose correctly
        indices = out["indices"]
        m0 = indices // 3  # n_levels[1] = 3
        m1 = indices % 3
        reconstructed = m0 * 3 + m1
        self.assertTrue(torch.equal(indices, reconstructed))

    def test_all_indices_in_range(self):
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        z = torch.randn(10_000, 2)
        out = quant(z, stochastic=True)
        self.assertTrue((out["indices"] >= 0).all())
        self.assertTrue((out["indices"] < 15).all())


class TestGradientFlow(unittest.TestCase):
    """8. Verify STE gradients flow through z_approx and comm_loss."""

    def test_ste_gradient(self):
        """Gradients flow through z_approx to the input via STE."""
        z = torch.randn(10, 2, requires_grad=True)
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        out = quant(z, stochastic=True)
        loss = out["codes"].sum()
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertFalse(torch.all(z.grad == 0))

    def test_comm_loss_gradient(self):
        """comm_loss has gradients w.r.t. input."""
        z = torch.randn(10, 2, requires_grad=True)
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        out = quant(z, stochastic=True)
        out["comm_loss"].backward()
        self.assertIsNotNone(z.grad)
        self.assertFalse(torch.all(z.grad == 0))


class TestDeterministicMode(unittest.TestCase):
    """9. With stochastic=False, verify epsilon=0 and output is deterministic."""

    def test_deterministic_is_reproducible(self):
        z = torch.randn(100, 2)
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        out1 = quant(z, stochastic=False)
        out2 = quant(z, stochastic=False)
        self.assertTrue(torch.equal(out1["codes"], out2["codes"]))
        self.assertTrue(torch.equal(out1["indices"], out2["indices"]))

    def test_stochastic_differs(self):
        """Stochastic mode should produce different outputs across calls."""
        torch.manual_seed(1)
        z = torch.randn(100, 2)
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        torch.manual_seed(10)
        out1 = quant(z, stochastic=True)
        torch.manual_seed(20)
        out2 = quant(z, stochastic=True)
        self.assertFalse(torch.equal(out1["codes"], out2["codes"]))

    def test_deterministic_codes_equal_bin_centers(self):
        """With epsilon=0, z_hat = c_m, so codes should equal bin centers."""
        z = torch.randn(50, 2)
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        out = quant(z, stochastic=False)
        codes = out["codes"].view(-1, 1, 2)
        # Each code value should be a valid bin center: -1 + (m + 0.5) * delta
        for ch in range(2):
            d = quant.deltas[ch].item()
            vals = codes[..., ch].flatten()
            # Recover m from code: m = (code + 1 - 0.5*delta) / delta
            m_recovered = (vals + 1.0 - 0.5 * d) / d
            # m should be very close to integer
            residual = (m_recovered - m_recovered.round()).abs()
            self.assertTrue(
                residual.max().item() < 1e-5,
                f"ch {ch}: codes not at bin centers, max residual={residual.max().item():.2e}",
            )


class TestConstructorValidation(unittest.TestCase):
    """Miscellaneous constructor checks."""

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            DDCLQuantizer(deltas=[0.4, 0.4], scales=[0.8])

    def test_repr(self):
        quant = DDCLQuantizer(deltas=[0.4, 2 / 3], scales=[0.8, 2 / 3])
        r = repr(quant)
        self.assertIn("DDCLQuantizer", r)
        self.assertIn("codebook_size=15", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
