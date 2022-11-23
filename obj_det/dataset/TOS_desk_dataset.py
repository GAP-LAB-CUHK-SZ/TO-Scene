import os
from torch.utils.data import Dataset
import numpy as np
from utils import pc_util
from data.model_utils_DOS import DOS_desk_config
from torch.utils.data import DataLoader
DC=DOS_desk_config()
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

type2class = {"bag":0, "bottle":1, "bowl":2, "camera":3, "can":4,
                            "cap":5, "clock":6, "keyboard":7, "display":8, "earphone":9,
                            "jar":10, "knife":11, "lamp":12, "laptop":13, "microphone":14,
                            "microwave":15, "mug":16, "printer":17, "remote control":18, "phone":19,
                            "alarm":20, "book":21, "cake":22, "calculator":23, "candle":24,
                            "charger":25, "chessboard":26, "coffee_machine":27, "comb":28, "cutting_board":29,
                            "dishes":30, "doll":31, "eraser":32, "eye_glasses":33, "file_box":34,
                            "fork":35, "fruit":36, "globe":37, "hat":38, "mirror":39,
                            "notebook":40, "pencil":41, "plant":42, "plate":43, "radio":44,
                            "ruler":45, "saucepan":46, "spoon":47, "tea_pot":48, "toaster":49,
                            "vase":50, "vegetables":51}
class2type = {type2class[t]:t for t in type2class}  # {0:'bag', ...}
nyu40ids = np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
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


class TOS_Desk_Dataset(Dataset):
    def __init__(self,config,mode):
        super(TOS_Desk_Dataset,self).__init__()
        self.config=config
        self.mode=mode
        self.split=os.listdir(os.path.join(self.config['data']['data_dir'],mode))

        '''filter some invalid samples'''
        if "easy" in self.config['data']['data_dir']:
            invalid_list=["id12052","id466","id8976","id2697","id7285","id6302","id7264","id3534","id8515","id3683",
                          "id10348","id4151","id4265","id7890","id3653","id11219","id844"]
        else:
            invalid_list = ["id1115", "id1249", "id3918", "id416"]
        new_split=[]
        for item in self.split:
            sample_id=item[0:-4]
            if sample_id not in invalid_list:
                new_split.append(sample_id)
        self.split=new_split
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
        sample_id=self.split[index]
        data_path = os.path.join(self.config['data']['data_dir'], self.mode ,sample_id + ".npz")
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
                point_votes_mask[ind] = 1.0
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
        ret_dict["id"]=sample_id
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

        return ret_dict

def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

def TOS_Desk_Dataloader(cfg,mode):
    dataset=TOS_Desk_Dataset(cfg,mode)
    dataloader=DataLoader(
        dataset=dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=(mode=="train"), num_workers=cfg['data']['num_workers'],
        drop_last=False, worker_init_fn=worker_init_fn
    )
    return dataloader


if __name__=="__main__":
    from Adaptive_Votenet.configs.config_utils import CONFIG
    cfg=CONFIG("./configs/train_heatmap.yaml").config
    dataset=TOS_Desk_Dataset(cfg,"train")
    for i in range(10):
        item=dataset.__getitem__(i)
        # for key in item:
        #     print(key)
        #     print(item[key])

