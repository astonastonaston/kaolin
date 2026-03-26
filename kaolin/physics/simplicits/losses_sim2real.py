# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import torch
import torch.nn as nn
from functools import partial
from kaolin.physics.materials import (
    material_utils,
    linear_elastic_material,
    neohookean_elastic_material)
from kaolin.physics.simplicits import weight_function_lbs
from kaolin.physics.utils.finite_diff import finite_diff_jac

__all__ = [
    'loss_ortho',
    'loss_elastic',
    'compute_losses'
]


def loss_ortho(weights):
    r"""Calculate orthogonality of weights

    Args:
        weights (torch.Tensor): Tensor of weights, of shape :math:`(\text{num_samples}, \text{num_handles})`

    Returns:
        torch.Tensor: Orthogonality loss, single value tensor
    """
    return nn.MSELoss()(weights.T @ weights, torch.eye(weights.shape[1], device=weights.device))


def loss_elastic(model, pts, yms, prs, rhos, transforms, appx_vol, interp_step, elasticity_type="neohookean", interp_material=False):
    r"""Calculate a version of simplicits elastic loss for training.

    Args:
        model (nn.Module): Simplicits object network
        pts (torch.Tensor): Tensor of sample points in :math:`\mathbb{R}^dim`, for now dim=3, of shape :math:`(\text{num_samples}, \text{dim})`
        yms (torch.Tensor): Length pt-wise youngs modulus, of shape :math:`(\text{num_samples})`
        prs (torch.Tensor): Length pt-wise poisson ratios, of shape :math:`(\text{num_samples})`
        rhos (torch.Tensor): Length pt-wise density, of shape :math:`(\text{num_samples})`
        transforms (torch.Tensor): Batch of sample transformations, of shape :math:`(\text{batch_size}, \text{num_handles}, \text{dim}, \text{dim}+1)`
        appx_vol (float): Approximate (or exact) volume of object (in :math:`m^3`)
        interp_step (float): Length interpolation schedule for neohookean elasticity (0%->100%)

    Returns:
        torch.Tensor: Elastic loss, single value tensor
    """

    mus, lams = material_utils.to_lame(yms, prs)

    partial_weight_fcn_lbs = partial(
        weight_function_lbs, tfms=transforms, fcn=model)
    pt_wise_Fs = finite_diff_jac(partial_weight_fcn_lbs, pts)
    pt_wise_Fs = pt_wise_Fs[:, :, 0]

    # shape (N, B, 3, 3)
    N, B = pt_wise_Fs.shape[0:2]

    # shape (N, B, 1)
    mus = mus.expand(N, B).unsqueeze(-1)
    lams = lams.expand(N, B).unsqueeze(-1)
    
    if interp_material:
        mus_min = mus.min()
        lams_min = lams.min()
        mus = (1 - interp_step) * mus_min + interp_step * mus
        lams = (1 - interp_step) * lams_min + interp_step * lams

    # ramps up from 100% linear elasticity to 100% neohookean elasticity
    # let the deformatio gradient (deformation difference) for each point satisfy material stiffness requirements (as weights in the loss of weights sums) for stability 
    # (i.e. avoid inverted elements) by using linear elasticity at the start of training, then gradually transition to neohookean elasticity
    lin_elastic = (1 - interp_step) * \
        linear_elastic_material._linear_elastic_energy(mus, lams, pt_wise_Fs)
    if elasticity_type == "neohookean":
        neo_elastic = (
            interp_step) * neohookean_elastic_material._neohookean_energy(mus, lams, pt_wise_Fs)
    else:
        raise ValueError(f"Elasticity type {elasticity_type} not supported")

    # weighted average (since we uniformly sample, this is uniform for now)
    return (appx_vol / pts.shape[0]) * (torch.sum(lin_elastic + neo_elastic))


def _lbs_deform_points(pts, weights, transforms):
    """Apply Simplicits-style LBS deformation.

    pts: (M,3) normalized points
    weights: (M,H) skinning weights at pts
    transforms: (H,3,4) handle transforms (delta-affine)
    returns: (M,3) deformed points
    """
    M = pts.shape[0]
    ones = torch.ones((M, 1), device=pts.device, dtype=pts.dtype)
    hom = torch.cat([pts, ones], dim=1)  # (M,4)
    # (M,H,3) = (M,4) x (H,4,3)
    deltas = torch.einsum('md,hde->mhe', hom, transforms.permute(0, 2, 1))
    disp = (deltas * weights.unsqueeze(-1)).sum(dim=1)
    return pts + disp


def _chamfer_l2(a, b):
    """Symmetric Chamfer (squared L2). a: (Na,3), b: (Nb,3)."""
    d = torch.cdist(a, b, p=2) ** 2  # (Na,Nb)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def compute_losses(
    model,
    normalized_pts,
    yms,
    prs,
    rhos,
    en_interp,
    batch_size,
    num_handles,
    appx_vol,
    num_samples,
    le_coeff,
    lo_coeff,
    # --- sim2real ---
    ls_coeff=0.0,
    sim2real_real_frames=None,
    sim2real_train_range=None,
    sim2real_num_pts_sim=2048,
    sim2real_num_pts_real=2048,
    sim2real_inner_steps=10,
    sim2real_inner_lr=5e-2,
    sim2real_reg=1e-4,
):
    r"""Perform a step of the Simplicits training process (+ optional sim2real Chamfer).

    Base (data-free) losses:
      - Elastic energy loss (physics prior)
      - Weight orthogonality loss

    Optional sim2real:
      - Sample a real point cloud frame Y_t
      - Fit handle transforms Z_t (inner loop) to match Y_t (weights detached)
      - Compute Chamfer(phi(X,Z_t), Y_t) and backprop to the weight network
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

    le = le_coeff * loss_elastic(model, sample_pts, sample_yms, sample_prs, sample_rhos,
                                 batch_transforms, appx_vol, en_interp)
    lo = lo_coeff * loss_ortho(weights)

    # -------------------------
    # Optional sim2real loss
    # -------------------------
    ls = torch.tensor(0.0, device=device, dtype=dtype)
    if (ls_coeff is not None) and (ls_coeff > 0.0) and (sim2real_real_frames is not None) and (sim2real_train_range is not None):
        start, end = sim2real_train_range
        end = min(end, len(sim2real_real_frames))
        start = max(0, start)
        if end > start:
            # Choose a frame in [start, end)
            t = int(torch.randint(low=start, high=end, size=(1,), device=device).item())
            real_pts = sim2real_real_frames[t].to(device=device, dtype=dtype)

            # Subsample real points
            if real_pts.shape[0] > sim2real_num_pts_real:
                ridx = torch.randint(0, real_pts.shape[0], (sim2real_num_pts_real,), device=device)
                real_pts_sub = real_pts[ridx]
            else:
                real_pts_sub = real_pts

            # Subsample sim points (rest samples)
            if normalized_pts.shape[0] > sim2real_num_pts_sim:
                sidx = torch.randint(0, normalized_pts.shape[0], (sim2real_num_pts_sim,), device=device)
                sim_pts = normalized_pts[sidx]
            else:
                sim_pts = normalized_pts

            # Fit Z with weights detached (stable; no bilevel 2nd-order)
            with torch.no_grad():
                w_det = model(sim_pts).detach()

            z = torch.zeros((num_handles, 3, 4), device=device, dtype=dtype, requires_grad=True)
            opt_z = torch.optim.Adam([z], lr=sim2real_inner_lr)

            for _ in range(int(sim2real_inner_steps)):
                opt_z.zero_grad(set_to_none=True)
                pred = _lbs_deform_points(sim_pts, w_det, z)
                lz = _chamfer_l2(pred, real_pts_sub) + sim2real_reg * (z ** 2).mean()
                lz.backward()
                opt_z.step()

            z_fit = z.detach()

            # Final chamfer for W update
            pred_final = _lbs_deform_points(sim_pts, model(sim_pts), z_fit)
            ls = ls_coeff * _chamfer_l2(pred_final, real_pts_sub)

    return le, lo, ls
