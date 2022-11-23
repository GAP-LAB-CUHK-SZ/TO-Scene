from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import sys

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import weighted_FPS._ext as _ext
except ImportError:
    print(ImportError)
    if not getattr(builtins, "__WEIGHTED_FPS_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *



class Weighted_FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint,alpha,t):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 4) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        fps_inds = _ext.furthest_point_sampling(xyz, npoint,alpha,t)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None


weighted_furthest_point_sample = Weighted_FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor
        idx : torch.Tensor
            (B, npoint) tensor of the features to gather
        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply