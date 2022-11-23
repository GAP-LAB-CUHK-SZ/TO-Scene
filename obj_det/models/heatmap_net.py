from external.pointnet2.pointnet2_modules import PointnetSAModuleVotes,PointnetFPModule
import torch
import torch.nn as nn


class Heatmap_Net(nn.Module):
    def __init__(self,input_feature_dim=1,npoints=[2048,1024,512,256],radius=[0.04,0.08,0.16,0.24]):
        super().__init__()
        print(input_feature_dim)
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

        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 128])
        self.fp3 = PointnetFPModule(mlp=[128+128,128,128])
        self.fp4 = PointnetFPModule(mlp=[128+input_feature_dim,64,32])
        self.out_conv=nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=32,kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32,out_channels=16,kernel_size=1),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16,out_channels=1,kernel_size=1),
            nn.Sigmoid()
        )
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, data_dict):
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
        end_points = {}
        pointcloud=data_dict["point_clouds"]
        batch_size = pointcloud.shape[0]
        #print(pointcloud.shape)

        xyz, features = self._break_up_pc(pointcloud)
        point_cloud_features=features
        point_cloud_xyz=xyz
        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['h_sa1_inds'] = fps_inds
        end_points['h_sa1_xyz'] = xyz
        end_points['h_sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['h_sa2_inds'] = fps_inds
        end_points['h_sa2_xyz'] = xyz
        end_points['h_sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['h_sa3_inds']=fps_inds
        end_points['h_sa3_xyz'] = xyz
        end_points['h_sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['h_sa4_inds'] = fps_inds
        end_points['h_sa4_xyz'] = xyz
        end_points['h_sa4_features'] = features

        # --------- 4 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['h_sa3_xyz'], end_points['h_sa4_xyz'], end_points['h_sa3_features'], end_points['h_sa4_features'])
        features = self.fp2(end_points['h_sa2_xyz'], end_points['h_sa3_xyz'], end_points['h_sa2_features'], features)
        features = self.fp3(end_points['h_sa1_xyz'], end_points['h_sa2_xyz'], end_points['h_sa1_features'], features)
        features = self.fp4(point_cloud_xyz, end_points['h_sa1_xyz'], point_cloud_features, features)
        out=self.out_conv(features)

        end_points["pred_heatmap"]=out

        gt_heatmap=data_dict["gt_heatmap"]
        #print(gt_heatmap.shape,out.transpose(1,2).shape)
        heatmap_loss=nn.MSELoss()(gt_heatmap,out.transpose(1,2))
        loss_dict={
            "heatmap_loss":heatmap_loss,
            "loss":heatmap_loss
        }

        return end_points,loss_dict

if __name__=="__main__":
    net=Heatmap_Net(input_feature_dim=1).cuda().float()
    point_cloud=torch.randn((2,40000,4)).cuda().float()
    out=net(point_cloud)
    print(out.shape)