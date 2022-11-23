import torch
import torch.nn as nn

from external.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from external.weighted_FPS.weighted_FPS_utils import weighted_furthest_point_sample as wFPS

class Pointnet2Backbone_adaptive(nn.Module):
    """
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0,npoints=[4096,2048,1024,512],radius=[0.04,0.08,0.16,0.24],alpha=2):
        super().__init__()
        self.alpha=alpha
        self.sa1 = PointnetSAModuleVotes(
            npoint=npoints[0],
            radius=radius[0],
            nsample=64,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=npoints[1],
            radius=radius[1],
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=npoints[2],
            radius=radius[2],
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=npoints[3],
            radius=radius[3],
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.fp1_fuse=PointnetFPModule(mlp=[128 + 128, 128])
        self.fp2_fuse = PointnetFPModule(mlp=[256 + 256, 256])
        self.fp3_fuse = PointnetFPModule(mlp=[256 + 256, 256])
        self.fp4_fuse = PointnetFPModule(mlp=[256 + 256, 256])

        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor,heatmap, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        #print(xyz.shape,heatmap.shape)
        heatmap=heatmap.transpose(1,2)
        fps_input = torch.cat([xyz, heatmap], dim=2)
        fps_inds = wFPS(fps_input, 4096, 1, self.alpha)
        #print("using weighted fps")
        xyz, features, _ = self.sa1(xyz, features,fps_inds.int())
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        h_features=end_points["h_sa1_features"]
        h_xyz=end_points["h_sa1_xyz"]
        features=self.fp1_fuse(xyz, h_xyz, features,h_features)

        heatmap=heatmap[torch.arange(batch_size)[:,None],fps_inds.long()]
        fps_input = torch.cat([xyz, heatmap], dim=2)
        fps_inds = wFPS(fps_input, 2048, 1, self.alpha)
        xyz, features, _ = self.sa2(xyz, features,fps_inds.int())  # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        h_features = end_points["h_sa2_features"]
        h_xyz = end_points["h_sa2_xyz"]
        features = self.fp2_fuse(xyz, h_xyz, features, h_features)

        heatmap = heatmap[torch.arange(batch_size)[:, None], fps_inds.long()]
        fps_input = torch.cat([xyz, heatmap], dim=2)
        fps_inds = wFPS(fps_input, 1024, 1, self.alpha)
        xyz, features, _ = self.sa3(xyz, features,fps_inds.int())  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        h_features = end_points["h_sa3_features"]
        h_xyz = end_points["h_sa3_xyz"]
        features = self.fp3_fuse(xyz, h_xyz, features, h_features)

        heatmap = heatmap[torch.arange(batch_size)[:, None], fps_inds.long()]
        fps_input = torch.cat([xyz, heatmap], dim=2)
        fps_inds = wFPS(fps_input, 512, 1, self.alpha)
        xyz, features, _ = self.sa4(xyz, features,fps_inds.int())  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        h_features = end_points["h_sa4_features"]
        h_xyz = end_points["h_sa4_xyz"]
        features = self.fp4_fuse(xyz, h_xyz, features, h_features)

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],features)
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][torch.arange(end_points['sa2_inds'].shape[0])[:,None], end_points['sa2_inds'].long()]  # indices among the entire input point clouds
        return end_points