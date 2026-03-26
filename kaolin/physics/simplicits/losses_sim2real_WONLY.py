# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Local patch: Sim2Real + optional material regularization for Simplicits training.

This module keeps the original Simplicits training objective:
  - Elastic energy loss (physics prior)
  - Orthogonality loss on weights

And adds:
  - Sim2Real supervised loss (Chamfer distance) between:
      simulated point cloud (from a tool-driven rollout) and a real reconstructed point cloud.

Design:
  - The simulator rollout is provided as a callable:
        sim2real_rollout_fn(model, t_idx) -> Tensor(P,3)
    The callable should return points in the SAME normalized coordinate frame as the real frames.

  - The rollout is treated as a black box (no autograd through Newton/Warp). Gradients flow
    through the returned simulated points ONLY if they were computed with differentiable ops
    using `model(...)`. (In our setup: we use solver to get transforms, then LBS forward pass
    is differentiable in W.)
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from kaolin.physics.materials import material_utils, linear_elastic_material, neohookean_elastic_material
from kaolin.physics.simplicits import weight_function_lbs
from kaolin.physics.utils.finite_diff import finite_diff_jac


__all__ = ["loss_ortho", "loss_elastic", "compute_losses"]


def loss_ortho(weights: torch.Tensor) -> torch.Tensor:
    """Orthogonality loss: || W^T W - I ||^2."""
    eye = torch.eye(weights.shape[1], device=weights.device, dtype=weights.dtype)
    return nn.MSELoss()(weights.T @ weights, eye)


def loss_elastic(
    model: nn.Module,
    pts: torch.Tensor,
    yms: torch.Tensor,
    prs: torch.Tensor,
    rhos: torch.Tensor,  # kept for signature consistency (not used directly)
    transforms: torch.Tensor,
    appx_vol: float,
    interp_step: float,
    elasticity_type: str = "neohookean",
    interp_material: bool = False,
) -> torch.Tensor:
    """Elastic energy loss used during weight training (matches Kaolin Simplicits)."""
    mus, lams = material_utils.to_lame(yms, prs)

    partial_weight_fcn_lbs = partial(weight_function_lbs, tfms=transforms, fcn=model)
    pt_wise_Fs = finite_diff_jac(partial_weight_fcn_lbs, pts)  # (N,B,1,3,3)
    pt_wise_Fs = pt_wise_Fs[:, :, 0]  # (N,B,3,3)

    n, b = pt_wise_Fs.shape[:2]
    mus = mus.expand(n, b).unsqueeze(-1)
    lams = lams.expand(n, b).unsqueeze(-1)

    if interp_material:
        mus_min = mus.min()
        lams_min = lams.min()
        mus = (1.0 - interp_step) * mus_min + interp_step * mus
        lams = (1.0 - interp_step) * lams_min + interp_step * lams

    lin_elastic = (1.0 - interp_step) * linear_elastic_material._linear_elastic_energy(mus, lams, pt_wise_Fs)
    if elasticity_type != "neohookean":
        raise ValueError(f"Elasticity type '{elasticity_type}' not supported")
    neo_elastic = interp_step * neohookean_elastic_material._neohookean_energy(mus, lams, pt_wise_Fs)

    return (appx_vol / pts.shape[0]) * torch.sum(lin_elastic + neo_elastic)


def _chamfer_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance with squared L2 metric."""
    d = torch.cdist(a, b, p=2) ** 2
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def compute_losses(
    model: nn.Module,
    normalized_pts: torch.Tensor,
    yms: torch.Tensor,
    prs: torch.Tensor,
    rhos: torch.Tensor,
    en_interp: float,
    batch_size: int,
    num_handles: int,
    appx_vol: float,
    num_samples: int,
    le_coeff: float,
    lo_coeff: float,
    # --- sim2real ---
    ls_coeff: float = 0.0,
    sim2real_real_frames: Optional[Sequence[torch.Tensor]] = None,
    sim2real_train_range: Optional[Tuple[int, int]] = None,
    sim2real_num_pts_sim: int = 2048,
    sim2real_num_pts_real: int = 2048,
    sim2real_rollout_fn: Optional[Callable[[nn.Module, int], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (le, lo, ls)."""
    device = normalized_pts.device
    dtype = normalized_pts.dtype

    # ---- base losses ----
    batch_transforms = 0.1 * torch.randn(batch_size, num_handles, 3, 4, dtype=dtype, device=device)

    sample_idx = torch.randint(0, normalized_pts.shape[0], (num_samples,), device=device)
    sample_pts = normalized_pts[sample_idx]
    sample_yms = yms[sample_idx]
    sample_prs = prs[sample_idx]
    sample_rhos = rhos[sample_idx]

    w = model(sample_pts)
    le = le_coeff * loss_elastic(model, sample_pts, sample_yms, sample_prs, sample_rhos, batch_transforms, appx_vol, en_interp)
    lo = lo_coeff * loss_ortho(w)

    # ---- sim2real ----
    ls = torch.tensor(0.0, device=device, dtype=dtype)

    if (
        ls_coeff is not None
        and ls_coeff > 0.0
        and sim2real_real_frames is not None
        and sim2real_train_range is not None
        and sim2real_rollout_fn is not None
    ):
        start, end = sim2real_train_range
        start = max(0, int(start))
        end = min(int(end), len(sim2real_real_frames))
        if end > start:
            t = int(torch.randint(low=start, high=end, size=(1,), device=device).item())

            real_pts = sim2real_real_frames[t].to(device=device, dtype=dtype)
            if real_pts.shape[0] > sim2real_num_pts_real:
                ridx = torch.randint(0, real_pts.shape[0], (sim2real_num_pts_real,), device=device)
                real_pts = real_pts[ridx]

            sim_pts = sim2real_rollout_fn(model, t).to(device=device, dtype=dtype)
            if sim_pts.shape[0] > sim2real_num_pts_sim:
                sidx = torch.randint(0, sim_pts.shape[0], (sim2real_num_pts_sim,), device=device)
                sim_pts = sim_pts[sidx]

            ls = ls_coeff * _chamfer_l2(sim_pts, real_pts)

    return le, lo, ls
