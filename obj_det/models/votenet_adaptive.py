# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from models.backbone_module import Pointnet2Backbone_adaptive
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
#from dump_helper import dump_results
from models.loss_helper import get_loss
from data.model_utils_DOS import DOS_desk_config,DOS_scene_config
from models.heatmap_net import Heatmap_Net
from utils.utils import load_checkpoint_model


class VoteNet_adaptive(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, cfg):
        super().__init__()
        if cfg['data']['dataset']=="TOS_desk":
            scene_level=False
        elif cfg['data']['dataset']=="TOS_scene":
            scene_level=True
        if scene_level:
            self.DC=DOS_scene_config()
        else:
            self.DC=DOS_desk_config()

        self.num_class = self.DC.num_class
        self.num_heading_bin = self.DC.num_heading_bin
        self.num_size_cluster = self.DC.num_size_cluster
        self.scene_level=scene_level
        self.mean_size_arr = self.DC.mean_size_arr
        assert (self.mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = cfg['model']['input_feature_dim']
        self.num_proposal = cfg['model']['num_proposal']
        self.vote_factor = cfg['model']['vote_factor']
        self.sampling = cfg['model']['sampling']

        # Backbone point feature learning
        '''
        Scene level and desk level have different parameters setting.
        '''
        if self.scene_level:
            self.heatmap_net = Heatmap_Net(input_feature_dim=self.input_feature_dim, npoints=[4096, 2048, 1024, 512],
                                           radius=[0.06, 0.25, 0.8, 1.2])
        else:
            self.heatmap_net = Heatmap_Net(input_feature_dim=self.input_feature_dim, npoints=[4096, 2048, 1024, 512],
                                           radius=[0.04, 0.08, 0.16, 0.24])
            '''pretrain heatmap prediction would help the performance in desktop setting'''
            pretrain_path = cfg['hm_pretrain_path']
            print("loading pretrain heatmap model from %s" % (pretrain_path))
            model_dict = load_checkpoint_model(pretrain_path)
            self.heatmap_net.load_state_dict(model_dict)
        if self.scene_level:
            self.backbone_net=Pointnet2Backbone_adaptive(input_feature_dim=self.input_feature_dim,radius=[0.06, 0.25, 0.8, 1.2],alpha=0.5)
        else:
            self.backbone_net =Pointnet2Backbone_adaptive(input_feature_dim=self.input_feature_dim,radius=[0.04, 0.08, 0.16, 0.24],alpha=2)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(self.num_class, self.num_heading_bin, self.num_size_cluster,
                                   self.mean_size_arr, self.num_proposal, self.sampling)

        self.criterion=get_loss

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]
        end_points, loss_dict = self.heatmap_net(inputs)
        pred_heatmap=end_points["pred_heatmap"]
        pred_heatmap=torch.clamp(pred_heatmap,min=0.1,max=1.0)
        end_points = self.backbone_net(inputs['point_clouds'], pred_heatmap,end_points)

        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points)

        for key in inputs:
            assert(key not in end_points)
            end_points[key]=inputs[key]

        loss,end_points=self.criterion(end_points,self.DC)
        '''scene and desktop setting has different loss ratio'''
        if self.scene_level:
            loss = loss + loss_dict["heatmap_loss"]
        else:
            loss = loss + 10 * loss_dict["heatmap_loss"]
        end_points["heatmap_loss"]=loss_dict["heatmap_loss"]

        return end_points,loss
