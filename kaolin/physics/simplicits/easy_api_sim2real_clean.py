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

import logging
import warnings
import warp as wp
import warp.sparse as wps
import numpy as np
import torch
import kaolin

from functools import partial

try:
    from losses_sim2real_clean import compute_losses  # local patched losses (clean)
except Exception:
    from kaolin.physics.simplicits.losses import compute_losses
from kaolin.physics.simplicits.network import SimplicitsMLP

import kaolin.physics.utils.warp_utilities as warp_utilities
import kaolin.physics.utils.torch_utilities as torch_utilities

from kaolin.physics.common import Collision, Gravity, Floor, Boundary
from kaolin.physics.materials import NeohookeanElasticMaterial
from kaolin.physics.materials.material_utils import get_defo_grad, to_lame
from kaolin.physics.common.optimization import newtons_method
from kaolin.physics.simplicits.precomputed import sparse_lbs_matrix, sparse_dFdz_matrix_from_dense
from kaolin.physics.simplicits.skinning import weight_function_lbs


logger = logging.getLogger(__name__)

__all__ = [
    'SimplicitsObject',
    'SimulatedObject',
    'SimplicitsScene',
]


class NormalizedSkinningWeightsFcn(torch.nn.Module):
    r"""
    A skinning weight function that normalizes the points to the unit cube and then applies the model, and appends a constant weight of 1.
    """
    def __init__(self, model, bb_min, bb_max):
        super().__init__()
        self.model = model
        self.bb_min = bb_min
        self.bb_max = bb_max

    def forward(self, pts):
        return torch.cat([
            self.model((pts - self.bb_min) / (self.bb_max - self.bb_min)),
            torch.ones((pts.shape[0], 1), device=pts.device)
        ], dim=1)


class SkinningWeightsFcn(torch.nn.Module):
    r"""
    A skinning weight function that applies the model to the points directly and appends a constant weight of 1.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pts):
        return torch.cat([
            self.model(pts),
            torch.ones((pts.shape[0], 1), device=pts.device)
        ], dim=1)


class SimplicitsObject:
    @staticmethod
    def create_trained(pts, yms, prs, rhos, appx_vol,
                       num_handles=10,
                       num_samples=1000,
                       model_layers=6,
                       training_batch_size=10,
                       training_num_steps=10000,
                       training_lr_start=1e-3,
                       training_lr_end=1e-3,
                       training_le_coeff=1e-1,
                       training_lo_coeff=1e6,
                       training_ls_coeff=0.0,
                       # --- optional: jointly learn material parameters ---
                       train_material_params: bool = False,
                       train_material_ym: bool = True,
                       train_material_pr: bool = True,
                       train_material_rho: bool = True,
                       # regularize learned material params toward init
                       material_reg_coeff: float = 0.0,
                       # --- sim2real supervision ---
                       sim2real_real_frames=None,
                       sim2real_train_range=None,
                       sim2real_num_pts_sim=2048,
                       sim2real_num_pts_real=2048,
                       # --- tool-driven rollout inputs ---
                       sim2real_tool_t=None,          # (T,3) normalized tool translations aligned to real_frames
                       sim2real_tool_R=None,          # (T,3,3) normalized rotations aligned to real_frames (optional)
                       sim2real_grasp_ids=None,       # (K,) indices into internal qp points (optional; auto-pick if None)
                       sim2real_grasp_k=10,           # K if auto-pick
                       sim2real_use_rotation=False,   # if True and sim2real_tool_R provided, rotate grasp patch
                       sim2real_grasp_penalty=1e5,    # penalty coefficient for grasp boundary
                       sim2real_scene_num_qp=1000,
                       sim2real_scene_timestep=0.01,
                       sim2real_scene_newton_steps=50,
                       sim2real_scene_direct_solve=True,
                       sim2real_max_steps_per_call=50,  # safety cap for rollout steps
                       training_log_every=1000,
                       normalize_for_training=True):
        r"""Constructs a SimplicitsObject by training a neural network to learn skinning weights.

        This method creates a SimplicitsObject by training a neural network to learn skinning weights
        that can be used for deformation. The network is trained to minimize a combination of
        local and global energy terms.
        
        Note:
            If num_handles is set to 0, the object will be created as rigid instead of deformable.
            The training process uses a combination of local and global energy terms to ensure
            both local detail preservation and global shape maintenance.

        Args:
            pts (torch.Tensor): Points tensor of shape :math:`(N, 3)` representing the object's geometry
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            appx_vol (torch.Tensor): Approximate volume tensor of shape :math:`(1,)`
            num_handles (int, optional): Number of control handles for deformation. Defaults to 10
            num_samples (int, optional): Number of samples used for training. Defaults to 1000
            model_layers (int, optional): Number of layers in the neural network. Defaults to 6
            training_batch_size (int, optional): Batch size for training. Defaults to 10
            training_num_steps (int, optional): Number of training iterations. Defaults to 10000
            training_lr_start (float, optional): Initial learning rate. Defaults to 1e-3
            training_lr_end (float, optional): Final learning rate. Defaults to 1e-3
            training_le_coeff (float, optional): Coefficient for local energy term. Defaults to 1e-1
            training_lo_coeff (float, optional): Coefficient for global energy term. Defaults to 1e6
            training_log_every (int, optional): Logging frequency during training. Defaults to 1000
            normalize_for_training (bool, optional): Whether to normalize points to unit cube for training. Defaults to True

        Returns:
            SimplicitsObject: A trained SimplicitsObject with learned skinning weights

        """
        if num_handles == 0:
            warnings.warn(
                f'Num Handles is 0. Simplicits Object will be created as rigid.', UserWarning)

            return SimplicitsObject.create_rigid(pts, yms, prs, rhos, appx_vol)
        
        if not torch.is_tensor(yms):
            yms = torch.full((pts.shape[0],), yms, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(prs):
            prs = torch.full((pts.shape[0],), prs, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(rhos):
            rhos = torch.full((pts.shape[0],), rhos, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(appx_vol):
            appx_vol = torch.tensor([appx_vol], dtype=pts.dtype, device=pts.device)

        device = pts.device

        bb_max = torch.max(pts, dim=0).values
        bb_min = torch.min(pts, dim=0).values
        bb_vol = (bb_max[0] - bb_min[0]) * (bb_max[1] -
                                            bb_min[1]) * (bb_max[2] - bb_min[2])

        # normalize the points
        if (normalize_for_training):
            # Normalize the appx vol of object
            norm_bb_max = torch.max((pts - bb_min) / (bb_max - bb_min),
                                    dim=0).values  # get the bb_max of the normalized pts
            norm_bb_min = torch.min((pts - bb_min) / (bb_max - bb_min),
                                    dim=0).values  # get the bb_min of the normalized pts

            norm_bb_vol = (norm_bb_max[0] - norm_bb_min[0]) * (norm_bb_max[1] -
                                                               norm_bb_min[1]) * (norm_bb_max[2] - norm_bb_min[2])
            normalized_pts = (pts - bb_min) / (bb_max - bb_min)
            norm_appx_vol = appx_vol * (norm_bb_vol / bb_vol)

            # Set pts, appx_vol, yms, prs, rhos to normalized values
            training_pts = normalized_pts
            training_appx_vol = norm_appx_vol
        else:
            training_pts = pts
            training_appx_vol = appx_vol

        training_yms = yms.unsqueeze(-1)
        training_prs = prs.unsqueeze(-1)
        training_rhos = rhos.unsqueeze(-1)

        # ------------------------------------------------------------
        # Optional: learn material parameters as GLOBAL modifiers.
        #
        # We keep the user-provided per-point fields (yms/prs/rhos) but
        # allow global adjustment driven by differentiable losses:
        #   yms'  = yms * exp(ym_log_scale)
        #   rhos' = rhos * exp(rho_log_scale)
        #   prs'  = clamp(prs + 0.48*tanh(pr_delta), (1e-4, 0.49))
        #
        # Differentiability:
        # - Elastic loss is differentiable w.r.t these parameters.
        # - Tool-driven sim2real rollout (Newton/Warp) is treated as a
        #   black box (no autograd), but we feed *current* materials into
        #   rollouts (detached) for train/eval consistency.
        # ------------------------------------------------------------
        base_yms = training_yms.detach()
        base_prs = training_prs.detach()
        base_rhos = training_rhos.detach()

        ym_log_scale = torch.nn.Parameter(
            torch.zeros((), device=device, dtype=pts.dtype),
            requires_grad=bool(train_material_params and train_material_ym),
        )
        pr_delta = torch.nn.Parameter(
            torch.zeros((), device=device, dtype=pts.dtype),
            requires_grad=bool(train_material_params and train_material_pr),
        )
        rho_log_scale = torch.nn.Parameter(
            torch.zeros((), device=device, dtype=pts.dtype),
            requires_grad=bool(train_material_params and train_material_rho),
        )

        def _current_material_tensors():
            ym_scale = torch.exp(ym_log_scale)
            rho_scale = torch.exp(rho_log_scale)

            yms_cur = base_yms * ym_scale
            rhos_cur = base_rhos * rho_scale

            pr_off = 0.48 * torch.tanh(pr_delta)
            prs_cur = torch.clamp(base_prs + pr_off, 1e-4, 0.49)
            return yms_cur, prs_cur, rhos_cur

        ym_log_scale_0 = ym_log_scale.detach().clone()
        pr_delta_0 = pr_delta.detach().clone()
        rho_log_scale_0 = rho_log_scale.detach().clone()

        ######### Train the model #########
        model = SimplicitsMLP(3, 64, num_handles, model_layers)  # input 3-dim vertex, output 5-dim handles
        model.to(device)

        opt_params = list(model.parameters())
        if train_material_params:
            yms_final, prs_final, rhos_final = _current_material_tensors()
            return SimplicitsObject(
                pts,
                yms_final.squeeze(-1).detach(),
                prs_final.squeeze(-1).detach(),
                rhos_final.squeeze(-1).detach(),
                appx_vol,
                skinning_weight_function,
            )

        return SimplicitsObject(pts, yms, prs, rhos, appx_vol, skinning_weight_function)

    @staticmethod
    def create_rigid(pts, yms, prs, rhos, appx_vol=1):
        r"""Creates a rigid SimplicitsObject with a single weight for affine deformations.

        This method creates a SimplicitsObject that behaves as a rigid body. At low stiffness values
        (young's modulus/ym), deformations will not be expressive, but with high stiffness values,
        the object will act as rigid.

        Args:
            pts (torch.Tensor): Points tensor of shape :math:`(N, 3)` representing the object's geometry
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            appx_vol (Union[torch.Tensor, float], optional): Approximate volume. Can be either:
                - A tensor of shape :math:`(1,)`
                - A float value. Defaults to 1

        Returns:
            SimplicitsObject: A rigid SimplicitsObject with a constant weight function
        """
        def constant_weight_function(x):
            return torch.ones(
                x.shape[0], 1, device=x.device, dtype=x.dtype)

        return SimplicitsObject.create_from_function(pts, yms, prs, rhos, appx_vol, constant_weight_function)

    @staticmethod
    def create_from_function(pts, yms, prs, rhos, appx_vol, fcn):
        r"""Creates a SimplicitsObject with a custom skinning weight function.

        This method creates a SimplicitsObject using a user-provided function to compute skinning weights.
        The function should take points as input and return a matrix of skinning weights.

        Args:
            pts (torch.Tensor): Points tensor of shape (N, 3) representing the object's geometry
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value
            appx_vol (Union[torch.Tensor, float]): Approximate volume. Can be either:
                - A tensor of shape (1,)
                - A float value
            fcn (callable): Function that takes points and returns skinning weights matrix

        Returns:
            SimplicitsObject: A SimplicitsObject with the provided skinning weight function
        """
        return SimplicitsObject(pts, yms, prs, rhos, appx_vol, skinning_weight_function=fcn)

    def __init__(self, pts, yms, prs, rhos, appx_vol, skinning_weight_function=None):
        r"""Initialize a SimplicitsObject with geometry, material properties, and skinning weights.

        A SimplicitsObject is a collection of points, material properties, and a linear blend skinning
        weight function that can be used to deform the object. Objects can be initialized in several ways
        using the static factory methods (read their docstrings for more details). Objects can also be
        denoted as kinematic or dynamic (default). Kinematic objects still have handles, but they are
        not solved for during simulation.

        Args:
            pts (torch.Tensor): Points tensor of shape (N, 3) representing the object's geometry
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            appx_vol (Union[torch.Tensor, float]): Approximate volume. Can be either:
                - A tensor of shape (1,)
                - A float value
            skinning_weight_function (callable, optional): Function that takes points and returns skinning weights matrix.
                If None, the object will be rigid. Defaults to None

        """
        if not torch.is_tensor(yms):
            yms = torch.full((pts.shape[0],), yms, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(prs):
            prs = torch.full((pts.shape[0],), prs, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(rhos):
            rhos = torch.full((pts.shape[0],), rhos, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(appx_vol):
            appx_vol = torch.tensor([appx_vol], dtype=pts.dtype, device=pts.device)

        self.pts = pts
        self.yms = yms
        self.prs = prs
        self.rhos = rhos
        self.appx_vol = appx_vol

        self.num_handles = skinning_weight_function(pts[:1]).shape[1]

        self.skinning_weight_function = skinning_weight_function

        self.device = pts.device
        self.dtype = pts.dtype

# The rest of the code remains unchanged.
