# Per-Channel DDCLQuantizer Redesign

**Date:** 2026-05-31
**Scope:** `dcmpc/utils/layers.py` (DDCLQuantizer), `dcmpc/dcmpc.py` (loss, metrics, soft CE), `dcmpc/config.py` (config classes)
**Goal:** Fix capacity mismatch between DDCL and FSQ by supporting per-channel delta/scale values, shifted floor quantization, and correct overflow prevention.

---

## Problem

The current DDCLQuantizer has three issues:

1. **Capacity mismatch:** A single `delta=1.0, scale=3.5` with `n_dims=2` produces 81 codes/group (9 levels per dim, joint 9^2), while FSQ [5,3] produces 15 codes/group. DDCL has 5.4x more nominal capacity, making R_m comparisons unfair.

2. **Ghost bins:** The floor formula `min_m = floor((-scale - delta/2) / delta)` creates one unreachable bin when `scale + delta/2` aligns to an integer multiple of delta.

3. **Single delta/scale for all channels:** FSQ [5,3] has different quantization resolution per channel (5 levels on ch0, 3 on ch1). DDCL uses a single delta for both, preventing matched capacity comparisons.

## Design

### Configuration

Replace scalar `ddcl_delta`, `ddcl_scale`, and `ddcl_n_dims` with per-channel lists:

```python
# In DCMPCConfig:
ddcl_deltas: List[float]   # per-channel bin widths
ddcl_scales: List[float]   # per-channel tanh pre-scaling factors
ddcl_lambda: float = 1e-3  # unchanged
```

`len(ddcl_deltas)` determines `num_channels`. Must equal `len(ddcl_scales)`.

**Derived quantities (inside DDCLQuantizer.__init__):**
- `n_levels_per_ch[i] = round(2 / deltas[i])` (number of bins per channel)
- `codebook_size = prod(n_levels_per_ch)`
- `offsets[i] = prod(n_levels_per_ch[i+1:])` (mixed-radix basis for joint index)

**Default values for matched-capacity comparison:**

| Task | FSQ levels | ddcl_deltas | ddcl_scales | Codes/group |
|------|-----------|-------------|-------------|-------------|
| Toy | [5, 3] | [0.4, 2/3] | [0.8, 2/3] | 15 |
| Transfer | [5, 5] | [0.4, 0.4] | [0.8, 0.8] | 25 |

The relationship `scale = 1 - delta/2` prevents dither overflow (see below), but scale is independently configurable for experimentation.

### DDCLQuantizer (layers.py)

**Constructor changes:**
- Accept `deltas: List[float]` and `scales: List[float]` instead of scalar `delta`, `scale`, `n_dims`
- `num_channels = len(deltas)`
- Compute `n_levels_per_ch`, `codebook_size`, `offsets` per channel
- Build per-channel center grids for the implicit codebook
- Register `deltas`, `scales`, `offsets`, `n_levels_per_ch` as buffers

**Forward pass changes:**

The quantization shifts from origin-centered to `[-1, 1)`-centered bins:

```python
# Per-channel operations (deltas/scales broadcast over [..., num_channels])
z_bounded = scales * tanh(z)                    # per-channel scale
epsilon = (rand_like - 0.5) * deltas            # per-channel dither width
z_prime = z_bounded + epsilon
m = floor((z_prime + 1) / deltas)               # shifted floor into [0, L) per channel
c_m = -1 + (m + 0.5) * deltas                   # bin center
z_hat = c_m - epsilon                            # dither cancellation
z_approx = z_bounded + (z_hat - z_bounded).detach()  # STE (unchanged)
```

**Overflow prevention:**
With `scale[ch] = 1 - delta[ch]/2`:
- `z_bounded ∈ (-scale, scale) = (-(1 - delta/2), 1 - delta/2)`
- `z_prime ∈ (-1, 1)` (open interval, since tanh never reaches ±scale)
- `(z_prime + 1) / delta ∈ (0, 2/delta) = (0, L)`
- `floor ∈ {0, 1, ..., L-1}` — exactly L bins, no overflow possible

No clamping, wrapping, or special-casing needed. The dither cancellation property `z_hat = z + independent_error` is preserved exactly.

**Comm loss:**
```python
comm_loss = lambda * log2(|z_bounded| / deltas + 1).mean()
```
Same formula, `deltas` broadcasts per-channel.

**Index computation:**
```python
# m is [batch, groups, num_channels], each channel in [0, n_levels_per_ch[ch]-1]
# offsets is [num_channels], mixed-radix: offsets[i] = prod(n_levels[i+1:])
indices = (m * offsets).sum(dim=-1)  # [batch, groups]
```

**Implicit codebook:**
Built via meshgrid over per-channel center lists (variable-length per channel).

### Loss function changes (dcmpc.py)

**Soft CE targets (ddcl_soft_ce_targets path):**
- Use per-channel `scales` and `deltas` instead of scalar `.scale` and `.delta`
- Use per-channel `n_levels_per_ch` instead of uniform `n_levels`
- Shift quantization: `m_det = floor((z_bounded_tar + 1) / deltas)`
- Fractional position: `f = (z_bounded_tar + 1) / deltas - m_det`
- `m_det` needs no clamping (overflow prevented by scale = 1 - delta/2)
- `m_adj` clamped per channel to `[0, n_levels_per_ch[ch] - 1]` at grid boundaries
- Mixed-radix offsets use `quantizer._offsets` (already per-channel)
- `codebook_size = quantizer.codebook_size` (now `prod(n_levels)`)

**Comm loss aggregation (loss method):**
No change — `comm_loss` is returned as a scalar from the quantizer and added to `aux_loss`.

### Metrics changes (dcmpc.py)

**`_build_metrics` DDCL section:**
```python
z_bounded = quantizer.scales * tanh(z["z"])  # per-channel scales
comms_bits = log2(|z_bounded| / quantizer.deltas + 1)  # per-channel deltas
```

**`_compute_rate_metrics`:**
- Same change for `allocated_bits` computation
- `codebook_size` from `quantizer.codebook_size` (already correct)
- `max_bits_per_group = log2(codebook_size)` (unchanged formula, correct value)

**`_empirical_entropy_bits`:**
Unchanged — operates on joint indices, which are already correctly computed.

### Config classes (config.py)

Update all DDCL config dataclasses:

```python
# Base DCMPCConfig
ddcl_deltas: List[float] = field(default_factory=lambda: [0.4, 2/3])
ddcl_scales: List[float] = field(default_factory=lambda: [0.8, 2/3])

# Remove: ddcl_delta, ddcl_scale, ddcl_n_dims
```

Each DDCL variant config class (DDCL_CE_Config, DDCL_MSE_Config, etc.) inherits these defaults. Transfer configs override with `[0.4, 0.4]` / `[0.8, 0.8]`.

### Backward compatibility

The old scalar fields `ddcl_delta`, `ddcl_scale`, `ddcl_n_dims` are removed. Existing Hydra configs from old runs will fail to load with the new code — this is acceptable since we need to rerun experiments anyway. The `measure_shared_dataset_rate.py` script will need minor updates to work with the new quantizer API.

### Tests

All tests in a new file `dcmpc/tests/test_ddcl_quantizer.py`:

1. **Bin count per channel:** For each `(delta, scale)` pair, verify exactly `round(2/delta)` unique m values are produced across a dense input sweep.

2. **No overflow:** With `scale = 1 - delta/2`, verify all indices are in `[0, codebook_size)` across 100k random samples with dither on.

3. **Dither cancellation:** Verify `E[z_hat - z_bounded] ≈ 0`, `Var[z_hat - z_bounded] ≈ delta^2/12`, and `Corr(z_bounded, z_hat - z_bounded) ≈ 0` for each channel.

4. **Codebook size matches FSQ:** For `ddcl_deltas=[0.4, 2/3]`, verify `codebook_size == 15`. For `[0.4, 0.4]`, verify `codebook_size == 25`.

5. **Per-channel independence:** With asymmetric deltas (e.g., [0.4, 2/3]), verify channel 0 has 5 levels and channel 1 has 3 levels.

6. **Comm loss uses per-channel delta:** Verify the comm_loss output changes appropriately when deltas differ per channel vs when they're uniform.

7. **Index computation:** Verify indices are valid mixed-radix encodings: `index = m[0] * n_levels[1] + m[1]` for 2-channel case.

8. **Gradient flow:** Verify gradients flow through `z_approx` to the encoder (STE works), and that `comm_loss` has gradients w.r.t. the input.

9. **Deterministic mode:** With `stochastic=False`, verify `epsilon=0` and output is deterministic.
