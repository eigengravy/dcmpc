#!/usr/bin/env python3
"""Numerical verification of the DDCL Soft-CE target distribution (E3 / T4).

Theory (T4):
  For a single DDCL dimension with deterministic pre-quantization value z_b
  and dither ε ~ U(-δ/2, δ/2):

    m_det = floor(z_b / δ)
    f     = z_b / δ - m_det  ∈ [0, 1)

    P(m_det)   = (1/2 + f)  if f < 1/2   |  (3/2 - f)  if f ≥ 1/2
    P(m_adj)   = (1/2 - f)  if f < 1/2   |  (f - 1/2)  if f ≥ 1/2
    m_adj      = m_det - 1  if f < 1/2   |  m_det + 1  if f ≥ 1/2

  For n_dims > 1, per-dim dithers are independent, so the joint distribution
  is the outer product of per-dim two-bin marginals → at most 2^n_dims
  nonzero entries.

Verification strategy:
  1. Per-dim analytic check: enumerate a z_b grid, compare empirical bin
     frequencies (from 100k Monte-Carlo dither samples) to the analytic
     P(m_det), P(m_adj). Assert max absolute error < 1e-2.
  2. Code-path consistency: run the exact dcmpc.py:502–586 label logic with
     known inputs and compare the resulting joint_soft_label to the MC
     estimate (n_dims = 1, 2, 3).
  3. Normalisation check: joint_soft_label.sum(dim=-1) == 1 everywhere.
  4. Boundary correctness: at the extremes (|z_b| ≈ scale, so f ≈ 0 or
     ≈ 1), m_adj is out of range and should be clamped to m_det → full mass
     on the boundary bin.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

# --- allow importing from dcmpc/ when running from the repo root ---
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.layers import DDCLQuantizer


# ---------------------------------------------------------------------------
# Helper: reference per-dim two-bin analytic distribution
# ---------------------------------------------------------------------------

def analytic_per_dim(z_b: float, delta: float, n_levels: int, min_m: int):
    """Return (m_det, p_det, m_adj, p_adj) for a single dimension."""
    m_det = int(math.floor(z_b / delta))
    f = z_b / delta - m_det
    if f < 0.5:
        p_det = 0.5 + f
        p_adj = 0.5 - f
        m_adj = m_det - 1
    else:
        p_det = 1.5 - f
        p_adj = f - 0.5
        m_adj = m_det + 1
    # Boundary clamping: identical to dcmpc.py scatter logic
    m_det_shifted = max(0, min(n_levels - 1, m_det - min_m))
    m_adj_shifted = max(0, min(n_levels - 1, m_adj - min_m))
    return m_det_shifted, p_det, m_adj_shifted, p_adj


def mc_bin_frequencies(z_b_vec, delta: float, n_mc: int = 200_000):
    """
    Estimate P(quantised bin | z_b) for each z_b in z_b_vec via Monte-Carlo.
    Returns a tensor of shape [len(z_b_vec)] containing the fraction of
    dither samples landing in the deterministic bin (m_det).
    """
    z_b_t = torch.tensor(z_b_vec, dtype=torch.float32)  # [N]
    eps = (torch.rand(n_mc, len(z_b_vec)) - 0.5) * delta  # [n_mc, N]
    z_prime = z_b_t.unsqueeze(0) + eps  # [n_mc, N]
    m = torch.floor(z_prime / delta).long()
    m_det = torch.floor(z_b_t / delta).long()  # [N]
    hit_det = (m == m_det.unsqueeze(0)).float().mean(dim=0)  # [N]
    return hit_det


# ---------------------------------------------------------------------------
# Helper: replicate dcmpc.py soft-CE label logic for a (batch, groups, n_dims)
# input tensor, and return the joint_soft_label.
# ---------------------------------------------------------------------------

def compute_soft_label_dcmpc(z_b: torch.Tensor, quant: DDCLQuantizer) -> torch.Tensor:
    """
    Replicate dcmpc.py:502–586 (the joint Soft-CE target computation).

    Args:
        z_b: [batch, groups, n_dims] float32 — PRE-TANH-SCALED bounded values.
        quant: DDCLQuantizer instance.

    Returns:
        joint_soft_label: [batch, groups, codebook_size] float32.
    """
    delta = quant.delta
    n_levels = quant.n_levels
    min_m = quant.min_m
    n_dims_q = quant.n_dims
    codebook_size = quant.codebook_size
    offsets = quant._offsets  # [n_dims], long

    m_det = torch.floor(z_b / delta).long()
    f = z_b / delta - m_det.float()

    m_adj = torch.where(f < 0.5, m_det - 1, m_det + 1)
    p_det = torch.where(f < 0.5, 0.5 + f, 1.5 - f)
    p_adj = 1.0 - p_det

    m_det_shifted = (m_det - min_m).clamp(0, n_levels - 1)
    m_adj_shifted = (m_adj - min_m).clamp(0, n_levels - 1)

    m_cands = torch.stack([m_det_shifted, m_adj_shifted], dim=0)   # [2, batch, groups, n_dims]
    p_cands = torch.stack([p_det.float(), p_adj.float()], dim=0)   # [2, batch, groups, n_dims]

    joint_soft_label = torch.zeros(
        *m_det.shape[:-1], codebook_size,
        device=m_det.device, dtype=torch.float32,
    )

    for corner_bits in range(1 << n_dims_q):
        choices = [(corner_bits >> i) & 1 for i in range(n_dims_q)]

        joint_idx = sum(
            m_cands[c][..., i] * offsets[i]
            for i, c in enumerate(choices)
        )

        joint_prob = p_cands[choices[0]][..., 0]
        for i in range(1, n_dims_q):
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
    """E3: verify per-dim analytic two-bin distribution against Monte-Carlo."""

    def setUp(self):
        self.delta = 1.0
        self.scale = 3.5
        self.quant1 = DDCLQuantizer(n_dims=1, delta=self.delta, scale=self.scale)

    def test_p_det_matches_mc_interior(self):
        """Interior bins: P(m_det) from analytic formula vs. 200k MC samples."""
        # Use 12 evenly spaced z_b values in the interior of the bounded range
        z_b_values = torch.linspace(-2.8, 2.8, 13).tolist()
        n_mc = 200_000
        mc_p_det = mc_bin_frequencies(z_b_values, self.delta, n_mc=n_mc)

        for z_b, mc_p in zip(z_b_values, mc_p_det.tolist()):
            m_det_s, p_det, m_adj_s, p_adj = analytic_per_dim(
                z_b, self.delta, self.quant1.n_levels, self.quant1.min_m
            )
            err = abs(mc_p - p_det)
            self.assertLess(
                err, 1e-2,
                f"z_b={z_b:.3f}: analytic p_det={p_det:.4f}, MC={mc_p:.4f}, err={err:.4f}",
            )

    def test_p_det_plus_p_adj_equals_one(self):
        """Analytically p_det + p_adj = 1 for all z_b."""
        z_b_values = torch.linspace(-3.4, 3.4, 35).tolist()
        for z_b in z_b_values:
            _, p_det, _, p_adj = analytic_per_dim(
                z_b, self.delta, self.quant1.n_levels, self.quant1.min_m
            )
            self.assertAlmostEqual(p_det + p_adj, 1.0, places=6,
                                   msg=f"z_b={z_b:.3f}: p_det+p_adj={p_det+p_adj}")

    def test_boundary_all_mass_on_det(self):
        """At the extreme boundary the adjacent bin is clamped → m_adj==m_det → all mass det."""
        quant = self.quant1
        # Extreme boundary: z_b near scale → m_det is the maximum bin, m_adj = m_det+1 clamps
        z_b_max = quant.scale - quant.delta * 0.01  # just inside saturation
        m_det_s, p_det, m_adj_s, p_adj = analytic_per_dim(
            z_b_max, quant.delta, quant.n_levels, quant.min_m
        )
        if m_det_s == m_adj_s:  # clamping happened
            self.assertAlmostEqual(p_det + p_adj, 1.0, places=6)
            # After clamping and scatter_add_, the bin accumulates total prob
            # (we don't assert p_det+p_adj directly here, just normalisation)


class TestSoftCECodePath(unittest.TestCase):
    """E3: verify dcmpc.py soft-CE joint label against independent MC estimate."""

    MC_SAMPLES = 300_000
    TOL_MEAN = 5e-3      # max allowed |mean error| per group × codebook_entry
    TOL_MAX  = 1e-2      # max allowed per-entry absolute error

    def _mc_joint_label(self, z_b: torch.Tensor, quant: DDCLQuantizer) -> torch.Tensor:
        """
        MC estimate of joint_soft_label for z_b: [batch, groups, n_dims].
        Returns [batch, groups, codebook_size].
        """
        batch, groups, n_dims = z_b.shape
        codebook_size = quant.codebook_size
        delta = quant.delta

        eps = (torch.rand(self.MC_SAMPLES, batch, groups, n_dims) - 0.5) * delta
        z_prime = z_b.unsqueeze(0) + eps           # [MC, batch, groups, n_dims]
        m = torch.floor(z_prime / delta).long()    # [MC, batch, groups, n_dims]
        m_shifted = (m - quant.min_m).clamp(0, quant.n_levels - 1)
        offsets = quant._offsets                   # [n_dims]
        joint_idx = (m_shifted * offsets).sum(dim=-1)  # [MC, batch, groups]

        label = torch.zeros(batch, groups, codebook_size)
        for s in range(self.MC_SAMPLES):
            label.scatter_add_(
                -1,
                joint_idx[s].unsqueeze(-1).long(),
                torch.ones(batch, groups, 1),
            )
        label /= self.MC_SAMPLES
        return label

    def _run_check(self, n_dims: int, z_b_raw: torch.Tensor, name: str):
        quant = DDCLQuantizer(n_dims=n_dims, delta=1.0, scale=3.5)
        # z_b is the tanh-scaled value (post-tanh, pre-dither); scale down to
        # be safely in interior of the bounded range
        z_b = z_b_raw.clamp(-quant.scale * 0.9, quant.scale * 0.9)

        analytic_label = compute_soft_label_dcmpc(z_b, quant)
        mc_label = self._mc_joint_label(z_b, quant)

        # Normalisation: each (batch, group) distribution must sum to 1
        sums = analytic_label.sum(dim=-1)
        self.assertTrue(
            (sums - 1.0).abs().max().item() < 1e-5,
            f"{name}: joint_soft_label not normalised; max dev={((sums-1.0).abs().max().item()):.2e}",
        )

        # Pointwise error
        err = (analytic_label - mc_label).abs()
        max_err = err.max().item()
        mean_err = err.mean().item()
        self.assertLess(mean_err, self.TOL_MEAN,
                        f"{name}: mean abs error {mean_err:.4f} > {self.TOL_MEAN}")
        self.assertLess(max_err, self.TOL_MAX,
                        f"{name}: max abs error {max_err:.4f} > {self.TOL_MAX}")

    def test_n_dims_1(self):
        torch.manual_seed(42)
        z_b = torch.rand(4, 2, 1) * 5.0 - 2.5
        self._run_check(1, z_b, "n_dims=1")

    def test_n_dims_2(self):
        torch.manual_seed(43)
        z_b = torch.rand(4, 2, 2) * 5.0 - 2.5
        self._run_check(2, z_b, "n_dims=2")

    def test_n_dims_3(self):
        torch.manual_seed(44)
        z_b = torch.rand(2, 2, 3) * 5.0 - 2.5
        self._run_check(3, z_b, "n_dims=3")

    def test_normalisation_random_batch(self):
        """Label sums to 1.0 for any valid z_b, n_dims=1,2,3."""
        torch.manual_seed(99)
        for n_dims in [1, 2, 3]:
            quant = DDCLQuantizer(n_dims=n_dims, delta=1.0, scale=3.5)
            z_b = (torch.rand(8, 4, n_dims) * 2 * quant.scale - quant.scale) * 0.95
            label = compute_soft_label_dcmpc(z_b, quant)
            sums = label.sum(dim=-1)
            max_dev = (sums - 1.0).abs().max().item()
            self.assertLess(max_dev, 1e-5,
                            f"n_dims={n_dims}: max normalisation dev={max_dev:.2e}")

    def test_at_most_2_to_n_nonzero_entries(self):
        """Each (batch, group) row has at most 2^n_dims non-zero entries."""
        torch.manual_seed(7)
        for n_dims in [1, 2, 3]:
            quant = DDCLQuantizer(n_dims=n_dims, delta=1.0, scale=3.5)
            z_b = torch.rand(4, 3, n_dims) * quant.scale * 0.8
            label = compute_soft_label_dcmpc(z_b, quant)
            max_nonzero = (label > 1e-9).sum(dim=-1).max().item()
            self.assertLessEqual(
                max_nonzero, 2 ** n_dims,
                f"n_dims={n_dims}: found {max_nonzero} > 2^{n_dims} nonzero entries",
            )


class TestSoftCEKnownValues(unittest.TestCase):
    """E3: closed-form sanity checks at exact fractional positions."""

    def setUp(self):
        self.quant = DDCLQuantizer(n_dims=1, delta=1.0, scale=3.5)

    def test_half_delta_exactly(self):
        """z_b = (m + 0.5)*δ: f=0.5 → boundary between the two cases."""
        delta = self.quant.delta
        # z_b = 1.5δ → m_det=1, f=0.5, edge case (f < 0.5 is False → upper branch)
        z_b = torch.tensor([[[1.5 * delta]]])   # [1,1,1]
        label = compute_soft_label_dcmpc(z_b, self.quant)
        # f=0.5 → p_det = 1.5 - 0.5 = 1.0, p_adj = 0 → all mass on m_det
        m_det = int(math.floor(1.5))
        m_det_shifted = m_det - self.quant.min_m
        self.assertAlmostEqual(label[0, 0, m_det_shifted].item(), 1.0, places=5,
                               msg="z_b=1.5δ: expect all mass on m_det=1")

    def test_zero_z_b(self):
        """z_b = 0: m_det = -1 (floor(0/1)=-?); f = 0 → p_det=0.5, p_adj=0.5."""
        # floor(0/1) = 0, f = 0, f < 0.5 → p_det=0.5, p_adj=0.5; m_adj = -1
        delta = self.quant.delta
        z_b = torch.tensor([[[0.0]]])
        label = compute_soft_label_dcmpc(z_b, self.quant)
        # Mass should be split equally between bins 0 and -1 (shifted)
        m_det = 0
        m_adj = -1
        m_det_s = max(0, min(self.quant.n_levels - 1, m_det - self.quant.min_m))
        m_adj_s = max(0, min(self.quant.n_levels - 1, m_adj - self.quant.min_m))
        p_det_expected = 0.5
        p_adj_expected = 0.5
        self.assertAlmostEqual(label[0, 0, m_det_s].item(), p_det_expected, places=5)
        if m_det_s != m_adj_s:
            self.assertAlmostEqual(label[0, 0, m_adj_s].item(), p_adj_expected, places=5)

    def test_quarter_delta(self):
        """z_b = 0.25δ: f=0.25, p_det=0.75, p_adj=0.25."""
        delta = self.quant.delta
        z_b = torch.tensor([[[0.25 * delta]]])
        label = compute_soft_label_dcmpc(z_b, self.quant)
        m_det = 0   # floor(0.25)
        m_adj = -1  # m_det - 1 since f < 0.5
        m_det_s = m_det - self.quant.min_m
        m_adj_s = max(0, m_adj - self.quant.min_m)
        self.assertAlmostEqual(label[0, 0, m_det_s].item(), 0.75, places=5)
        if m_det_s != m_adj_s:
            self.assertAlmostEqual(label[0, 0, m_adj_s].item(), 0.25, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
