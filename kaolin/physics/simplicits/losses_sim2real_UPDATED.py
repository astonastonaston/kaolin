# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local patch: add an optional Sim2Real loss for Simplicits weight training.

This file keeps the original Simplicits training losses (elastic + orthogonality),
and adds a *supervised* term that compares a *tool-driven* simulated point cloud
against a real reconstructed point cloud via Chamfer distance.

Key design choice (important for stability):
- The physics solver is run outside this module (in easy_api_sim2real.py) and
  provided here as a Python callable `sim2real_rollout_fn(model, t_idx) -> (P,3)`.
- This lets you choose how expensive the rollout is (num steps, num_qp, etc.)
  without coupling the loss implementation to the simulator internals.
"""

import torch
import torch.nn as nn
from functools import partial

from kaolin.physics.materials import (
    material_utils,
    linear_elastic_material,
    neohookean_elastic_material,
)
from kaolin.physics.simplicits import weight_function_lbs
from kaolin.physics.utils.finite_diff import finite_diff_jac

__all__ = [
    "loss_ortho",
    "loss_elastic",
    "compute_losses",
]

# -------------------------
# Base Simplicits losses
# -------------------------

def loss_ortho(weights: torch.Tensor) -> torch.Tensor:
    """Orthogonality loss: || W^T W - I ||^2"""
    return nn.MSELoss()(weights.T @ weights, torch.eye(weights.shape[1], device=weights.device))


def loss_elastic(
    model: nn.Module,
    pts: torch.Tensor,
    yms: torch.Tensor,
    prs: torch.Tensor,
    rhos: torch.Tensor,
    transforms: torch.Tensor,
    appx_vol: float,
    interp_step: float,
    elasticity_type: str = "neohookean",
    interp_material: bool = False,
) -> torch.Tensor:
    """Simplicits elastic energy loss used during weight training."""
    mus, lams = material_utils.to_lame(yms, prs)

    partial_weight_fcn_lbs = partial(weight_function_lbs, tfms=transforms, fcn=model)
    pt_wise_Fs = finite_diff_jac(partial_weight_fcn_lbs, pts)
    pt_wise_Fs = pt_wise_Fs[:, :, 0]  # (N,B,3,3)

    N, B = pt_wise_Fs.shape[0:2]
    mus = mus.expand(N, B).unsqueeze(-1)
    lams = lams.expand(N, B).unsqueeze(-1)

    if interp_material:
        mus_min = mus.min()
        lams_min = lams.min()
        mus = (1 - interp_step) * mus_min + interp_step * mus
        lams = (1 - interp_step) * lams_min + interp_step * lams

    lin_elastic = (1 - interp_step) * linear_elastic_material._linear_elastic_energy(mus, lams, pt_wise_Fs)
    if elasticity_type != "neohookean":
        raise ValueError(f"Elasticity type {elasticity_type} not supported")
    neo_elastic = interp_step * neohookean_elastic_material._neohookean_energy(mus, lams, pt_wise_Fs)

    return (appx_vol / pts.shape[0]) * (torch.sum(lin_elastic + neo_elastic))


# -------------------------
# Sim2Real helpers
# -------------------------

def _chamfer_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer (squared L2).

    a: (Na,3), b: (Nb,3)
    """
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
    sim2real_real_frames=None,
    sim2real_train_range=None,
    sim2real_num_pts_sim: int = 2048,
    sim2real_num_pts_real: int = 2048,
    sim2real_rollout_fn=None,
):
    """Compute (le, lo, ls).

    sim2real_rollout_fn:
        Callable(model, t_idx) -> torch.Tensor (P,3) in *same normalized space*
        as sim2real_real_frames[t_idx].
    """
    device = normalized_pts.device
    dtype = normalized_pts.dtype

    # -------------------------
    # Base (data-free) losses
    # -------------------------
    batch_transforms = 0.1 * torch.randn(batch_size, num_handles, 3, 4, dtype=dtype, device=device)

    sample_indices = torch.randint(low=0, high=normalized_pts.shape[0], size=(num_samples,), device=device)
    sample_pts = normalized_pts[sample_indices]
    sample_yms = yms[sample_indices]
    sample_prs = prs[sample_indices]
    sample_rhos = rhos[sample_indices]

    weights = model(sample_pts)

    le = le_coeff * loss_elastic(
        model, sample_pts, sample_yms, sample_prs, sample_rhos, batch_transforms, appx_vol, en_interp
    )
    lo = lo_coeff * loss_ortho(weights)

    # -------------------------
    # Optional sim2real loss
    # -------------------------
    ls = torch.tensor(0.0, device=device, dtype=dtype)

    if (
        (ls_coeff is not None)
        and (ls_coeff > 0.0)
        and (sim2real_real_frames is not None)
        and (sim2real_train_range is not None)
        and (sim2real_rollout_fn is not None)
    ):
        start, end = sim2real_train_range
        end = min(end, len(sim2real_real_frames))
        start = max(0, start)

        if end > start:
            t = int(torch.randint(low=start, high=end, size=(1,), device=device).item())

            real_pts = sim2real_real_frames[t].to(device=device, dtype=dtype)
            if real_pts.shape[0] > sim2real_num_pts_real:
                ridx = torch.randint(0, real_pts.shape[0], (sim2real_num_pts_real,), device=device)
                real_pts = real_pts[ridx]

            # Run simulator rollout (tool-driven) to get simulated point cloud at frame t
            sim_pts = sim2real_rollout_fn(model, t).to(device=device, dtype=dtype)
            if sim_pts.shape[0] > sim2real_num_pts_sim:
                sidx = torch.randint(0, sim_pts.shape[0], (sim2real_num_pts_sim,), device=device)
                sim_pts = sim_pts[sidx]

            ls = ls_coeff * _chamfer_l2(sim_pts, real_pts)

    return le, lo, ls
