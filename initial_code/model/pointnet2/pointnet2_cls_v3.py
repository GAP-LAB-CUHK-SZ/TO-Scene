from collections import namedtuple

import torch
import torch.nn as nn

from model.pointnet2.pointnet2_modules import PointNet2SAModule, PointNet2SAModuleMSG
from util import pt_util


class PointNet2SSGCls(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Classification network
        Parameters
        ----------
        k: int
            Number of semantics classes to predict over -- size of softmax classifier
        c: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """
    def __init__(self, c=3, k=40, use_xyz=True):
        super(PointNet2SSGCls, self).__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointNet2SAModule(npoint=512, radius=None, nsample=32, mlp=[c, 64, 64, 128], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=128, radius=None, nsample=64, mlp=[128, 128, 128, 256], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(mlp=[256, 256, 512, 1024], use_xyz=use_xyz))
        self.FC_layer = nn.Sequential(pt_util.FC(1024, 512, bn=True), nn.Dropout(p=0.5), pt_util.FC(512, 256, bn=True), nn.Dropout(p=0.5), pt_util.FC(256, k, activation=None))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + c) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.FC_layer(features.squeeze(-1))


class PointNet2MSGCls(PointNet2SSGCls):
    r"""
        PointNet2 with multi-scale grouping
        Classification network

        Parameters
        ----------
        k: int
            Number of semantics classes to predict over -- size of softmax classifier
        c: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, c=3, k=40, use_xyz=True, sync_bn=False):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointNet2SAModuleMSG(npoint=512, radii=[0.1, 0.2, 0.4], nsamples=[16, 32, 128], mlps=[[c, 32, 32, 64], [c, 64, 64, 128], [c, 64, 96, 128]], use_xyz=use_xyz))
        c = 64 + 128 + 128
        self.SA_modules.append(PointNet2SAModuleMSG(npoint=128, radii=[0.2, 0.4, 0.8], nsamples=[32, 64, 128], mlps=[[c, 64, 64, 128], [c, 128, 128, 256], [c, 128, 128, 256]], use_xyz=use_xyz))
        c = 128 + 256 + 256
        self.SA_modules.append(PointNet2SAModule(mlp=[c, 256, 512, 1024], use_xyz=use_xyz))
        self.FC_layer = nn.Sequential(pt_util.FC(1024, 512, bn=True), nn.Dropout(p=0.5), pt_util.FC(512, 256, bn=True), nn.Dropout(p=0.5), pt_util.FC(256, k, activation=None))
        if sync_bn:
            convert_to_syncbn(self)


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            preds = model(inputs)
            labels = labels.view(-1)
            loss = criterion(preds, labels)
            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()
            return ModelReturn(preds, loss, {"acc": acc.item(), 'loss': loss.item()})
    return model_fn


if __name__ == "__main__":
    import numpy as np
    import torch.optim as optim
    B, N = 2, 2048
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()
    model = PointNet2SSGCls(c=3, k=3)
    print(model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    print("testing with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()

    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()
    model = PointNet2SSGCls(c=3, k=3, use_xyz=False)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    print("Testing without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()

    model = PointNet2MSGCls(c=3, k=3, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    print("testing with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()

    # With with use_xyz=False
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()
    model = PointNet2MSGCls(c=3, k=3, use_xyz=False)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    print("testing without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()