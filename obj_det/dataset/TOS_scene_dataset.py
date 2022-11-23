import os
from torch.utils.data import Dataset
import numpy as np
from Adaptive_Votenet.utils import pc_util
from data.scannet_dense_desktop.model_utils_scannet_scene import ScannetDatasetConfig_scene
DC=ScannetDatasetConfig_scene()
MAX_NUM_OBJ = 156
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
                            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
                            'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17,
                            "bag":18, "bottle":19, "bowl":20, "camera":21, "can":22,
                            "cap":23, "clock":24, "keyboard":25, "display":26, "earphone":27,
                            "jar":28, "knife":29, "lamp":30, "laptop":31, "microphone":32,
                            "microwave":33, "mug":34, "printer":35, "remote control":36, "phone":37,
                            "alarm":38, "book":39, "cake":40, "calculator":41, "candle":42,
                            "charger":43, "chessboard":44, "coffee_machine":45, "comb":46, "cutting_board":47,
                            "dishes":48, "doll":49, "eraser":50, "eye_glasses":51, "file_box":52,
                            "fork":53, "fruit":54, "globe":55, "hat":56, "mirror":57,
                            "notebook":58, "pencil":59, "plant":60, "plate":61, "radio":62,
                            "ruler":63, "saucepan":64, "spoon":65, "tea_pot":66, "toaster":67,
                            "vase":68, "vegetables":69}     # 把68 umbrella去掉了
class2type = {type2class[t]:t for t in type2class}  # {0:'bag', ...}
nyu40ids = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39,
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93])
small_object_nyu40ids=np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93])

def rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))

    dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))

    for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:, 0] = crnr[0] * dx
        crnrs[:, 1] = crnr[1] * dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:, i] = crnrs[:, 0]
        new_y[:, i] = crnrs[:, 1]

    new_dx = 2.0 * np.max(new_x, 1)
    new_dy = 2.0 * np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)

class TOS_Scene_Dataset(Dataset):
    def __init__(self,config,mode):
        super(TOS_Scene_Dataset,self).__init__()
        self.config=config.config
        self.mode=mode
        split_path=os.path.join(self.config['data']['split_dir'],mode+"_split_scenelevel-augment-relabel.txt")

        invalid_list = ["scene0644_00_07","scene0644_00_07","scene0632_00_00","scene0395_00_02","scene0574_01_00",
                        "scene0468_02_00","scene0621_00_01","scene0656_02_02","scene0644_00_01","scene0575_00_00","scene0700_02_00"]
        with open(split_path,'r') as f:
            org_split_list=f.readlines()
            split_list=[]
            for item in org_split_list:
                split_id=item.rstrip("\n")
                if split_id not in invalid_list:
                    split_list.append(split_id)
        self.split=split_list
        self.use_color=self.config['data']['use_color']
        self.use_height=self.config['data']['use_height']
        if mode=="train":
            self.augment=self.config['data']['use_aug']
        else:
            self.augment=False
        self.npoints=self.config['data']['npoints']

    def __len__(self):
        return len(self.split)

    def __getitem__(self,index):
        id=self.split[index]
        data_path=os.path.join(self.config['data']['data_dir'],id+".npz")
        data_content=np.load(data_path)
        xyz=data_content['xyz']
        color=data_content['color']
        bbox=data_content['bbox']
        heatmap=data_content['heatmap']
        instance_label=data_content['instance_label']
        semantic_label=data_content['semantic_label']
        if not self.use_color:
            point_cloud=xyz
        else:
            point_cloud=np.concatenate([xyz,color],axis=1)
            point_cloud[:,3:]=(point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0

        if self.use_height:
            floor_height=np.percentile(point_cloud[:,2],0.99)
            height=point_cloud[:,2]-floor_height
            point_cloud=np.concatenate([point_cloud,np.expand_dims(height,1)],axis=1)

        # -------------------------------- LABELS ----------------------------------------
        target_bboxes=np.zeros((MAX_NUM_OBJ,6))
        target_bboxes_mask=np.zeros((MAX_NUM_OBJ))
        angle_classes=np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))


        if point_cloud.shape[0]>self.npoints:
            random_choice = np.random.choice(point_cloud.shape[0], self.npoints, replace=False)
        else:
            random_choice = np.random.choice(point_cloud.shape[0], self.npoints, replace=True)

        point_cloud=point_cloud[random_choice]
        ### TODO: add gt votes
        heatmap=heatmap[random_choice]
        instance_label=instance_label[random_choice]
        semantic_label=semantic_label[random_choice]



        target_bboxes_mask[0:bbox.shape[0]] = 1
        target_bboxes[0:bbox.shape[0], :] = bbox[:, 0:6]

        if self.augment:
            if np.random.random()>0.5:
                point_cloud[:,0]=-point_cloud[:,0]
                target_bboxes[:,0]=-target_bboxes[:,0]
            if np.random.random()>0.5:
                point_cloud[:,1]=-point_cloud[:,1]
                target_bboxes[:, 1] = -target_bboxes[:, 1]

            rot_angle=(np.random.random()*np.pi/18)-np.pi/36 #-5 ~+5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes=rotate_aligned_boxes(target_bboxes,rot_mat)

        point_votes = np.zeros([point_cloud.shape[0], 3])
        point_votes_mask = np.zeros(point_cloud.shape[0])
        for i_instance in np.unique(instance_label):
            # find all points belong to that instance
            ind = np.where(instance_label == i_instance)[0]
            # find the semantic label
            if semantic_label[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                if semantic_label[ind[0]] in small_object_nyu40ids:
                    point_votes_mask[ind] = 1
                else:
                    point_votes_mask[ind]=1
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in bbox[:, -1]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:bbox.shape[0]] = class_ind
        size_residuals[0:bbox.shape[0], :] = \
            target_bboxes[0:bbox.shape[0], 3:6] - DC.mean_size_arr[class_ind, :]

        target_bboxes[target_bboxes_mask == 0, :] = -10000

        ret_dict={}
        ret_dict["point_clouds"]=point_cloud.astype(np.float32)
        ret_dict["gt_heatmap"]=heatmap.astype(np.float32)
        ret_dict["id"]=id
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        ret_dict['gt_bbox']=target_bboxes.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bbox.shape[0]] = \
            [DC.nyu40id2class[x] for x in bbox[:, -1][0:bbox.shape[0]]]
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['sem_cls_label'] = size_classes.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict["pcd_color"]=color[random_choice].astype(np.uint8)

        return ret_dict