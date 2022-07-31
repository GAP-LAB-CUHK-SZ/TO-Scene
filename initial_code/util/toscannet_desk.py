import os
import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import sa_create, collate_fn
from util.data_util import data_prepare_v101 as data_prepare

# from voxelize import voxelize
# from data_util_check import sa_create, collate_fn
# from data_util_check import data_prepare_v101 as data_prepare

# Map relevant classes to {0,1,...,53}, 0 to unannoted class
remapper = np.ones(300) * (255)
# for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
#     remapper[x] = i
for i, x in enumerate([41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]):
    remapper[x] = i

class TOScanNet(Dataset):
    def __init__(self, split='train', data_root='trainval', split_root = "list", voxel_size=0.04, voxel_max=None, transform=None, 
                 shuffle_index=False, loop=1, repeat_align=False):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop, self.repeat_align = \
            split, voxel_size, transform, voxel_max, shuffle_index, loop, repeat_align
        
        # get the data_list
        data_list = os.path.join(split_root, "%s_split.txt"%(split))
        with open(data_list) as dl:
            data_list = dl.read().splitlines()
        self.data_list = [item[13:-6] for item in data_list]
        # print(self.data_list)
        
        # get data
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npz')
                f = np.load(data_path)  # npz: xyz, color, semantic_label, instance_label
                data = np.concatenate((f['xyz'], f['color'], np.expand_dims(f['semantic_label'], axis=-1), np.expand_dims(f['instance_label'], axis=-1)), axis=-1)  # npy, n*8
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))

        # nan = []
        # inf = []
        # total_num_pts = []
        # obj_num_pts = []
        # for item in self.data_list:
        #     data = SA.attach("shm://{}".format(item)).copy()
        #     coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        #     # total_num_pts.append(coord.shape[0])
        #     label = remapper[label.astype(int)]
        #     keep_idx = np.where((label != 255.))
        #     coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
        #     if coord.shape[0] == 0:
        #         exit("!!!!!!")
        #     obj_num_pts.append(coord.shape[0])
        # # #     nan.append(torch.isnan(torch.from_numpy(data)).any())
        # # #     inf.append(torch.isinf(torch.from_numpy(data)).any())
        # print("!!!!!total:", sum(total_num_pts))
        # print("!!!!!obj:", sum(obj_num_pts))
        # exit(0)
        # for item in nan:
        #     if item == torch.tensor(True):
        #         print("!")
        #         exit(0)
        # for item in inf:
        #     if item == torch.tensor(True):
        #         print("!!")
        #         exit(0)

        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        label = remapper[label.astype(int)]
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index, self.repeat_align)
        
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop


if __name__ == '__main__':
    data_root = '/remote-home/chenpei/data/output/process_5e6/npz2-desklevel-recrop'
    split_root = '/remote-home/chenpei/data/output/meta_data'
    voxel_size, voxel_max = 0.004, 8000

    point_data = TOScanNet(split='train', data_root=data_root, split_root=split_root, voxel_size=voxel_size, voxel_max=voxel_max)
    print('point data size:', point_data.__len__())

    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    for idx in range(1):
        end = time.time()
        voxel_num = []
        for i, (coord, feat, label, offset) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            print('tag', coord.shape, feat.shape, label.shape, offset.shape, torch.unique(label))
            if torch.isnan(coord).any() or torch.isnan(feat).any() or torch.isnan(label).any():
                print("!")
                exit(0)
            if torch.isinf(coord).any() or torch.isinf(feat).any() or torch.isinf(label).any():
                print("!")
                exit(0)   
            voxel_num.append(label.shape[0])
            end = time.time()
    print(np.sort(np.array(voxel_num)))
