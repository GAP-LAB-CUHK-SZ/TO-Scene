from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops
from util import pt_util


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Transformer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, radius=None, nsample=32, use_softmax=True):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 4
        self.conv_q = nn.Conv1d(in_planes, mid_planes, kernel_size=1)
        self.conv_k = nn.Conv1d(in_planes, mid_planes, kernel_size=1)
        self.conv_v = nn.Conv1d(in_planes, out_planes, kernel_size=1)
        self.share_planes, self.nsample = share_planes, nsample
        self.grouper = pointops.QueryAndGroup(radius, nsample)
        self.conv_p = nn.Conv2d(3, mid_planes, kernel_size=1)
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, p, x) -> torch.Tensor:
        x_q, x_k, x_v = self.conv_q(x), self.conv_k(x), self.conv_v(x)  # (B, C, N)
        B, C, N = x_v.size()
        p_t = p.transpose(1, 2).contiguous()  # (B, N, 3)
        x_k = self.grouper(p_t, p_t, x_k)  # (B, 3+C, N, nsample)
        x_v = self.grouper(p_t, p_t, x_v)  # (B, 3+C, N, nsample)
        p_r = self.conv_p(x_k[:, 0:3, :, :]) # (B, C, N, nsample)
        x_k, x_v = x_k[:, 3:, :, :], x_v[:, 3:, :, :]  # (B, C, N, nsample)
        w = (x_q.unsqueeze(-1) * x_k + p_r).view(B, -1, C // self.share_planes, N, self.nsample).sum(1)  # (B, C // self.share_planes, N, nsample)
        if self.use_softmax:
            w = self.softmax(w)
        x = (x_v.view(B, self.share_planes, C // self.share_planes, N, self.nsample) * w.unsqueeze(1)).view(B, C, N, self.nsample).sum(-1)
        return x


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, radius=None, nsample=32):
        super().__init__()
        self.conv = conv1x1(in_planes, out_planes)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if self.stride != 1:
            self.grouper = pointops.QueryAndGroup(radius, nsample, use_xyz=False)
            self.pool = nn.MaxPool2d((1, nsample))

    def forward(self, p, x) -> torch.Tensor:
        # p, x = px[0], px[1]
        x = self.relu(self.bn(self.conv(x)))
        if self.stride != 1:
            B, C, N = x.size()
            # M = N // self.stride  # new centroids
            M = self.stride
            p_t = p.transpose(1, 2).contiguous()  # (B, N, 3)
            idx = pointops.furthestsampling(p_t, M)  # (B, M)
            p_n = pointops.gathering(p, idx)  # (B, 3, M)
            p_n_t = p_n.transpose(1, 2).contiguous()  # (B, M, 3)
            x = self.grouper(p_t, p_n_t, x)  # (B, 3+C, M, nsample)
            x = self.pool(x).squeeze(-1)  # (B, C, M)
            p = p_n
        return p, x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, radius=None, nsample=32):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = Transformer(planes, planes, share_planes, radius, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, p, x):
        # p, x = px[0], px[1]
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(p, x)))
        x = self.bn3(self.conv3(x))
        # x += identity
        x = self.relu(x)
        return x


class _PointWebSAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.transition = None
        self.bottleneck = None

    def forward(self, p: torch.Tensor, x: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        p = p.transpose(1, 2).contiguous()
        p, x = self.transition(p, x)
        for i in range(0, len(self.bottleneck)):
            x = self.bottleneck[i](p, x)
        p = p.transpose(1, 2).contiguous()
        return p, x


class PointWebSAModule(_PointWebSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    nsample : int32
        Number of sample
    mlps : list of int32
        Spec of the MLP before the global max_pool
    mlps2: list of list of int32
        Spec of the MLP for AFA
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, npoint: int = None, nsample: int = None, mlp: List[int] = None, mlp2: List[int] = None, bn: bool = True, use_xyz: bool = True, use_bn = True):
        super().__init__()
        '''
        self.npoint = npoint
        self.grouper = pointops.QueryAndGroup(nsample=nsample, use_xyz=use_xyz) if npoint is not None else pointops.GroupAll(use_xyz)
        if use_xyz:
            mlp[0] += 3
        if npoint is not None:
            mlp_tmp = pt_util.SharedMLP([mlp[0]] + mlp2, bn=use_bn)
            mlp_tmp.add_module('weight', (pt_util.SharedMLP([mlp2[-1], mlp[0]], bn=False, activation=None)))
            self.afa = _AFAModule(mlp=mlp_tmp)
        self.mlp = pt_util.SharedMLP(mlp, bn=bn)
        '''
        self.transition = Transition(mlp[0], mlp[1], stride=npoint, radius=None, nsample=nsample)
        self.bottleneck = nn.ModuleList()
        for i in range(1, len(mlp)-1):
            self.bottleneck.append(Bottleneck(mlp[i], mlp[i+1], share_planes=8, radius=None, nsample=nsample))


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    c = 6
    xyz = torch.randn(2, 8, 3, requires_grad=True).cuda()
    xyz_feats = torch.randn(2, 8, c, requires_grad=True).cuda()

    test_module = PointWebSAModule(npoint=2, nsample=6, mlp=[c, 32, 32], mlp2=[16, 16], use_bn=True)
    test_module.cuda()
    xyz_feats = xyz_feats.transpose(1, 2).contiguous()
    print(test_module)
    print(test_module(xyz, xyz_feats))

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
