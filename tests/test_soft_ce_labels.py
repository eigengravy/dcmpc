#!/usr/bin/env python3
"""Numerical verification of the DDCL Soft-CE target distribution.

Theory:
  For a single DDCL dimension with deterministic pre-quantization value z_b
  and dither ε ~ U(-δ/2, δ/2), using shifted floor m = floor((z_b+1)/δ):

    f = (z_b + 1) / δ - m  ∈ [0, 1)

    P(m_det)   = (1/2 + f)  if f < 1/2   |  (3/2 - f)  if f ≥ 1/2
    P(m_adj)   = (1/2 - f)  if f < 1/2   |  (f - 1/2)  if f ≥ 1/2
    m_adj      = m_det - 1  if f < 1/2   |  m_det + 1  if f ≥ 1/2

  For num_channels > 1, per-dim dithers are independent, so the joint
  distribution is the outer product of per-dim two-bin marginals → at most
  2^num_channels nonzero entries.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.layers import DDCLQuantizer


# ---------------------------------------------------------------------------
# Helper: reference per-dim two-bin analytic distribution (shifted floor)
# ---------------------------------------------------------------------------

def analytic_per_dim(z_b: float, delta: float, n_levels: int):
    """Return (m_det, p_det, m_adj, p_adj) for a single dimension using shifted floor."""
    m_det = int(math.floor((z_b + 1.0) / delta))
    f = (z_b + 1.0) / delta - m_det
    if f < 0.5:
        p_det = 0.5 + f
        p_adj = 0.5 - f
        m_adj = m_det - 1
    else:
        p_det = 1.5 - f
        p_adj = f - 0.5
        m_adj = m_det + 1
    m_det = max(0, min(n_levels - 1, m_det))
    m_adj = max(0, min(n_levels - 1, m_adj))
    return m_det, p_det, m_adj, p_adj


def mc_bin_frequencies(z_b_vec, delta: float, n_mc: int = 200_000):
    """MC estimate of P(m_det | z_b) for each z_b using shifted floor."""
    z_b_t = torch.tensor(z_b_vec, dtype=torch.float32)
    eps = (torch.rand(n_mc, len(z_b_vec)) - 0.5) * delta
    z_prime = z_b_t.unsqueeze(0) + eps
    m = torch.floor((z_prime + 1.0) / delta).long()
    m_det = torch.floor((z_b_t + 1.0) / delta).long()
    hit_det = (m == m_det.unsqueeze(0)).float().mean(dim=0)
    return hit_det


# ---------------------------------------------------------------------------
# Helper: replicate dcmpc.py soft-CE label logic for new per-channel API
# ---------------------------------------------------------------------------

def compute_soft_label_dcmpc(z_b: torch.Tensor, quant: DDCLQuantizer) -> torch.Tensor:
    """
    Replicate dcmpc.py soft-CE joint label computation.

    Args:
        z_b: [..., num_channels] float32 — PRE-TANH-SCALED bounded values.
        quant: DDCLQuantizer instance.

    Returns:
        joint_soft_label: [..., codebook_size] float32.
    """
    deltas = quant.deltas
    n_levels_per_ch = quant.n_levels_per_ch
    num_ch = quant.num_channels
    codebook_size = quant.codebook_size
    offsets = quant._offsets

    m_det = torch.floor((z_b + 1.0) / deltas).long()
    f = (z_b + 1.0) / deltas - m_det.float()

    m_adj = torch.where(f < 0.5, m_det - 1, m_det + 1)
    p_det = torch.where(f < 0.5, 0.5 + f, 1.5 - f)
    p_adj = 1.0 - p_det

    # m_det is in [0, L-1] by construction; m_adj may be -1 or L at boundaries
    max_bin = (n_levels_per_ch - 1)
    # broadcast max_bin to match z_b dims: add leading singleton dims
    for _ in range(z_b.ndim - 1):
        max_bin = max_bin.unsqueeze(0)
    m_det_shifted = m_det
    m_adj_shifted = torch.clamp(m_adj, min=0).min(max_bin)

    m_cands = torch.stack([m_det_shifted, m_adj_shifted], dim=0)
    p_cands = torch.stack([p_det.float(), p_adj.float()], dim=0)

    joint_soft_label = torch.zeros(
        *m_det.shape[:-1], codebook_size,
        device=m_det.device, dtype=torch.float32,
    )

    for corner_bits in range(1 << num_ch):
        choices = [(corner_bits >> i) & 1 for i in range(num_ch)]

        joint_idx = sum(
            m_cands[c][..., i] * offsets[i]
            for i, c in enumerate(choices)
        )

        joint_prob = p_cands[choices[0]][..., 0]
        for i in range(1, num_ch):
            joint_prob = joint_prob * p_cands[choices[i]][..., i]

        joint_soft_label.scatter_add_(
            -1,
            joint_idx.unsqueeze(-1).long(),
            joint_prob.unsqueeze(-1),
        )

    return joint_soft_label


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSoftCEAnalyticVsMC(unittest.TestCase):
    """Verify per-dim analytic two-bin distribution against Monte-Carlo."""

    def setUp(self):
        self.delta = 0.4
        self.scale = 0.8
        self.quant1 = DDCLQuantizer(deltas=[self.delta], scales=[self.scale])

    def test_p_det_matches_mc_interior(self):
        """Interior bins: P(m_det) from analytic formula vs. 200k MC samples."""
        n_levels = self.quant1._n_levels_per_ch[0]
        z_b_values = torch.linspace(-0.6, 0.6, 13).tolist()
        n_mc = 200_000
        mc_p_det = mc_bin_frequencies(z_b_values, self.delta, n_mc=n_mc)

        for z_b, mc_p in zip(z_b_values, mc_p_det.tolist()):
            m_det_s, p_det, m_adj_s, p_adj = analytic_per_dim(
                z_b, self.delta, n_levels
            )
            err = abs(mc_p - p_det)
            self.assertLess(
                err, 1e-2,
                f"z_b={z_b:.3f}: analytic p_det={p_det:.4f}, MC={mc_p:.4f}, err={err:.4f}",
            )

    def test_p_det_plus_p_adj_equals_one(self):
        """Analytically p_det + p_adj = 1 for all z_b."""
        n_levels = self.quant1._n_levels_per_ch[0]
        z_b_values = torch.linspace(-0.7, 0.7, 35).tolist()
        for z_b in z_b_values:
            _, p_det, _, p_adj = analytic_per_dim(
                z_b, self.delta, n_levels
            )
            self.assertAlmostEqual(p_det + p_adj, 1.0, places=6,
                                   msg=f"z_b={z_b:.3f}: p_det+p_adj={p_det+p_adj}")


class TestSoftCECodePath(unittest.TestCase):
    """Verify dcmpc.py soft-CE joint label against independent MC estimate."""

    MC_SAMPLES = 300_000
    TOL_MEAN = 5e-3
    TOL_MAX  = 1e-2

    def _mc_joint_label(self, z_b: torch.Tensor, quant: DDCLQuantizer) -> torch.Tensor:
        """MC estimate of joint_soft_label for z_b: [batch, groups, num_ch]."""
        batch, groups, num_ch = z_b.shape
        codebook_size = quant.codebook_size
        deltas = quant.deltas

        eps = (torch.rand(self.MC_SAMPLES, batch, groups, num_ch) - 0.5) * deltas
        z_prime = z_b.unsqueeze(0) + eps
        m = torch.floor((z_prime + 1.0) / deltas).long()
        # clamp per channel
        max_bin = (quant.n_levels_per_ch - 1)[None, None, None, :]
        m = torch.clamp(m, min=0).min(max_bin)
        offsets = quant._offsets
        joint_idx = (m * offsets).sum(dim=-1)

        label = torch.zeros(batch, groups, codebook_size)
        for s in range(self.MC_SAMPLES):
            label.scatter_add_(
                -1,
                joint_idx[s].unsqueeze(-1).long(),
                torch.ones(batch, groups, 1),
            )
        label /= self.MC_SAMPLES
        return label

    def _run_check(self, deltas, scales, z_b_raw: torch.Tensor, name: str):
        quant = DDCLQuantizer(deltas=deltas, scales=scales)
        # keep z_b safely in interior
        max_scale = min(scales) * 0.9
        z_b = z_b_raw.clamp(-max_scale, max_scale)

        analytic_label = compute_soft_label_dcmpc(z_b, quant)
        mc_label = self._mc_joint_label(z_b, quant)

        sums = analytic_label.sum(dim=-1)
        self.assertTrue(
            (sums - 1.0).abs().max().item() < 1e-5,
            f"{name}: joint_soft_label not normalised; max dev={((sums-1.0).abs().max().item()):.2e}",
        )

        err = (analytic_label - mc_label).abs()
        max_err = err.max().item()
        mean_err = err.mean().item()
        self.assertLess(mean_err, self.TOL_MEAN,
                        f"{name}: mean abs error {mean_err:.4f} > {self.TOL_MEAN}")
        self.assertLess(max_err, self.TOL_MAX,
                        f"{name}: max abs error {max_err:.4f} > {self.TOL_MAX}")

    def test_1ch(self):
        torch.manual_seed(42)
        z_b = torch.rand(4, 2, 1) * 1.0 - 0.5
        self._run_check([0.4], [0.8], z_b, "1ch")

    def test_2ch_symmetric(self):
        torch.manual_seed(43)
        z_b = torch.rand(4, 2, 2) * 1.0 - 0.5
        self._run_check([0.4, 0.4], [0.8, 0.8], z_b, "2ch_symmetric")

    def test_2ch_asymmetric(self):
        torch.manual_seed(44)
        z_b = torch.rand(4, 2, 2) * 1.0 - 0.5
        self._run_check([0.4, 2/3], [0.8, 2/3], z_b, "2ch_asymmetric")

    def test_normalisation_random_batch(self):
        """Label sums to 1.0 for various configs."""
        torch.manual_seed(99)
        configs = [
            ([0.4], [0.8]),
            ([0.4, 0.4], [0.8, 0.8]),
            ([0.4, 2/3], [0.8, 2/3]),
        ]
        for deltas, scales in configs:
            quant = DDCLQuantizer(deltas=deltas, scales=scales)
            num_ch = len(deltas)
            z_b = (torch.rand(8, 4, num_ch) * 2 * min(scales) - min(scales)) * 0.95
            label = compute_soft_label_dcmpc(z_b, quant)
            sums = label.sum(dim=-1)
            max_dev = (sums - 1.0).abs().max().item()
            self.assertLess(max_dev, 1e-5,
                            f"deltas={deltas}: max normalisation dev={max_dev:.2e}")

    def test_at_most_2_to_n_nonzero_entries(self):
        """Each row has at most 2^num_channels non-zero entries."""
        torch.manual_seed(7)
        configs = [
            ([0.4], [0.8]),
            ([0.4, 0.4], [0.8, 0.8]),
            ([0.4, 2/3], [0.8, 2/3]),
        ]
        for deltas, scales in configs:
            quant = DDCLQuantizer(deltas=deltas, scales=scales)
            num_ch = len(deltas)
            z_b = torch.rand(4, 3, num_ch) * min(scales) * 0.8
            label = compute_soft_label_dcmpc(z_b, quant)
            max_nonzero = (label > 1e-9).sum(dim=-1).max().item()
            self.assertLessEqual(
                max_nonzero, 2 ** num_ch,
                f"deltas={deltas}: found {max_nonzero} > 2^{num_ch} nonzero entries",
            )


class TestSoftCEKnownValues(unittest.TestCase):
    """Closed-form sanity checks at exact fractional positions."""

    def setUp(self):
        self.quant = DDCLQuantizer(deltas=[0.4], scales=[0.8])

    def test_zero_z_b(self):
        """z_b = 0: shifted floor m = floor((0+1)/0.4) = floor(2.5) = 2."""
        z_b = torch.tensor([[[0.0]]])
        label = compute_soft_label_dcmpc(z_b, self.quant)
        # (0+1)/0.4 = 2.5, m_det=2, f=0.5 → upper branch: p_det=1.0
        self.assertAlmostEqual(label[0, 0, 2].item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
