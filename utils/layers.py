#!/usr/bin/env python3
import copy
import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state
from vector_quantize_pytorch import FSQ as _FSQ
from vector_quantize_pytorch import VectorQuantize as _VQ

from .helper import orthogonal_init


def mlp(
    in_dim: int,
    mlp_dims: List[int],
    out_dim: int,
    act_fn: Optional[Callable] = None,
    dropout: float = 0.0,
):
    """
    MLP with LayerNorm, Mish activations, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]

    dims = [int(in_dim)] + mlp_dims + [int(out_dim)]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(
        NormedLinear(dims[-2], dims[-1], act=act_fn)
        if act_fn
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


def mlp_ensemble(in_dim: int, mlp_dims: List[int], out_dim: int, size: int):
    """Ensemble of MLPs with orthogonal initialization"""
    mlp_list = []
    for _ in range(size):
        mlp_list.append(mlp(in_dim, mlp_dims, out_dim))
        orthogonal_init(mlp_list[-1].parameters())
    return Ensemble(mlp_list)


class FSQ(_FSQ):
    """
    Finite Scalar Quantization
    """

    def __init__(self, levels: List[int]):
        super().__init__(levels=levels)
        self.levels = levels
        self.num_channels = len(levels)

    def forward(self, z):
        shp = z.shape
        z = z.view(*shp[:-1], -1, self.num_channels)
        if z.ndim > 3:  # TODO this might not work for CNN
            codes, indices = torch.func.vmap(super().forward)(z)
        else:
            codes, indices = super().forward(z)
        codes = codes.flatten(-2)
        return {"codes": codes, "indices": indices, "z": z, "state": codes}

    def __repr__(self):
        return f"FSQ(levels={self.levels})"


class DDCLQuantizer(nn.Module):
    """
    DDCL (Differentiable Discrete Communication Learning) quantizer
    adapted for 1D latent vectors.

    Uses tanh bounding, uniform dithering, floor quantization, and STE.
    Matches FSQ's interface so it can be swapped in as self._quantizer.

    Reference: https://arxiv.org/abs/2511.01554
    """

    def __init__(
        self,
        n_dims: int = 2,
        delta: float = 1.0,
        scale: float = 3.5,
        ddcl_lambda: float = 1e-3,
    ):
        super().__init__()
        self.n_dims = n_dims
        self.num_channels = n_dims
        self.delta = delta
        self.scale = scale
        self.ddcl_lambda = ddcl_lambda

        half_delta = delta / 2.0
        min_m = math.floor((-scale - half_delta) / delta)
        max_m = math.floor((scale + half_delta) / delta)
        self.n_levels = max_m - min_m + 1
        self._codebook_size = self.n_levels**n_dims
        self.min_m = min_m

        offsets = [self.n_levels ** (n_dims - 1 - i) for i in range(n_dims)]
        self.register_buffer("_offsets", torch.tensor(offsets, dtype=torch.long))

        centers = torch.tensor(
            [delta * (m + 0.5) for m in range(min_m, min_m + self.n_levels)]
        )
        grids = torch.meshgrid(*([centers] * n_dims), indexing="ij")
        codebook = torch.stack([g.flatten() for g in grids], dim=-1)
        self.register_buffer("_implicit_codebook", codebook)

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def implicit_codebook(self) -> torch.Tensor:
        return self._implicit_codebook

    def forward(self, z):
        shp = z.shape
        z = z.view(*shp[:-1], -1, self.num_channels)

        z_bounded = self.scale * torch.tanh(z)

        if self.training:
            epsilon = (torch.rand_like(z_bounded) - 0.5) * self.delta
            z_prime = z_bounded + epsilon
            m = torch.floor(z_prime / self.delta)
            c_m = self.delta * (m + 0.5)
            z_hat = c_m - epsilon
            z_approx = z_bounded + (z_hat - z_bounded).detach()
        else:
            m = torch.round(z_bounded / self.delta)
            z_approx = self.delta * (m + 0.5)

        comm_loss = self.ddcl_lambda * torch.log2(
            z_bounded.abs() / self.delta + 1.0
        ).mean()

        m_shifted = (m.long() - self.min_m).clamp(0, self.n_levels - 1)
        indices = (m_shifted * self._offsets).sum(dim=-1)

        codes = z_approx.flatten(-2)
        return {
            "codes": codes,
            "indices": indices,
            "z": z,
            "state": codes,
            "comm_loss": comm_loss,
        }

    def __repr__(self):
        return (
            f"DDCLQuantizer(n_dims={self.n_dims}, delta={self.delta}, "
            f"scale={self.scale}, n_levels={self.n_levels}, "
            f"codebook_size={self.codebook_size})"
        )


class VQQuantizer(nn.Module):
    """
    Vector Quantization wrapper using vector_quantize_pytorch.VectorQuantize.

    Splits 1D latent into groups and quantizes each group against a shared
    learned codebook. Matches FSQ's interface for use as self._quantizer.
    """

    def __init__(self, codebook_size: int = 15, codebook_dim: int = 2):
        super().__init__()
        self.num_channels = codebook_dim
        self._codebook_size = codebook_size
        self._vq = _VQ(
            dim=codebook_dim,
            codebook_size=codebook_size,
            accept_image_fmap=False,
        )

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def implicit_codebook(self) -> torch.Tensor:
        return self._vq.codebook

    def forward(self, z):
        shp = z.shape
        z_grouped = z.view(*shp[:-1], -1, self.num_channels)

        orig_shape = z_grouped.shape
        z_flat = z_grouped.reshape(-1, z_grouped.shape[-2], self.num_channels)

        z_q, indices, commit_loss = self._vq(z_flat)

        z_q = z_q.reshape(orig_shape)
        indices = indices.reshape(orig_shape[:-1])

        codes = z_q.flatten(-2)
        return {
            "codes": codes,
            "indices": indices,
            "z": z_grouped,
            "state": codes,
            "commit_loss": commit_loss,
        }

    def __repr__(self):
        return (
            f"VQQuantizer(codebook_size={self.codebook_size}, "
            f"codebook_dim={self.num_channels})"
        )


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, Mish activation, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None
        self.norm = nn.LayerNorm(self.out_features)

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)

        x = self.norm(x)
        x = self.act(x)
        return x

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, \
        out_features={self.out_features}, \
        bias={self.bias is not None}{repr_dropout}, \
        act={self.act.__class__.__name__})"


class Ensemble(nn.Module):
    """Vectorized ensemble of modules"""

    def __init__(self, modules, **kwargs):
        super().__init__()
        self.params_dict, self._buffers = stack_module_state(modules)
        self.params = nn.ParameterList([p for p in self.params_dict.values()])
        self.size = len(modules)

        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        base_model = copy.deepcopy(modules[0])
        base_model = base_model.to("meta")

        def fmodel(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))

        self.vmap = torch.vmap(
            fmodel, in_dims=(0, 0, None), randomness="different", **kwargs
        )
        self._repr = str(nn.ModuleList(modules))

    def forward(self, *args, **kwargs):
        return self.vmap(self._get_params_dict(), self._buffers, *args, **kwargs)

    def _get_params_dict(self):
        params_dict = {}
        for key, value in zip(self.params_dict.keys(), self.params):
            params_dict.update({key: value})
        return params_dict

    def __repr__(self):
        return f"Vectorized " + self._repr
