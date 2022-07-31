import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize
# from voxelize import voxelize

# from ply import write_ply

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def collate_fn_bin(batch):
    coord, feat, label, label_bin = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.cat(label_bin), torch.IntTensor(offset)


def area_crop(coord, area_rate, split='train'):
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= coord_min; coord_max -= coord_min
    x_max, y_max = coord_max[0:2]
    x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
    if split == 'train' or split == 'trainval':
        x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
    else:
        x_s, y_s = (x_max - x_size) / 2, (y_max - y_size) / 2
    x_e, y_e = x_s + x_size, y_s + y_size
    crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
    return crop_idx


def data_dup(xyz, color, label, num_out):
    num_in = label.shape[0]
    dup = np.random.choice(num_in, num_out - num_in)
    xyz_dup = xyz[dup, ...]
    xyz_dup = np.concatenate([xyz, xyz_dup], 0)
    color_dup = color[dup, ...]
    color_dup = np.concatenate([color, color_dup], 0)
    label = np.expand_dims(label, axis=-1)
    label_dup = label[dup]
    label_dup = np.concatenate([label, label_dup], 0)
    label_dup = label_dup.squeeze(-1)
    return xyz_dup, color_dup, label_dup


def load_kitti_data(data_path):
    data = np.fromfile(data_path, dtype=np.float32)
    data = data.reshape((-1, 4))  # xyz+remission
    return data


def load_kitti_label(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape(-1)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def data_prepare_gpc(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None,
                     shuffle_index=False, fea_dim=6, fp_norm=False):
    # if transform:
    #     coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    # apply normalization augmentations:
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    centered_points = np.copy(coord)
    centered_points[:, :2] = coord[:, :2] - (coord_max[:2] + coord_min[:2]) / 2.0
    if fp_norm:
        normalized_points = (coord - coord_min) / (coord_max - coord_min)
    else:
        normalized_points = coord / coord_max
    normalized_colors = feat / 255.0

    if transform is not None:
        centered_points, normalized_colors = transform(centered_points, normalized_colors)

    # current feat
    if fea_dim == 3:
        current_feat = normalized_colors
    elif fea_dim == 4:
        current_feat = np.concatenate((normalized_colors, normalized_points[:, 2:]), axis=-1)
    elif fea_dim == 5:
        current_feat = np.concatenate((normalized_colors, normalized_points[:, 2:], np.ones((centered_points.shape[0], 1))), axis=-1)
    elif fea_dim == 6:
        current_feat = np.concatenate((normalized_colors, normalized_points), axis=-1)

    # to Tensor
    centered_points = torch.FloatTensor(centered_points)
    current_feat = torch.FloatTensor(current_feat)
    current_labels = torch.LongTensor(label)

    return centered_points, current_feat, current_labels


def data_prepare_v101(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
    # # print(label.shape)
    # keep_idx = np.where((label != 255.))
    # coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
    # print("!", label.shape)
    
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        # print("voxelized", label.shape)
        # # visualize
        # field_names = ['x', 'y', 'z', 'red', 'green', 'blue','values']
        # # write_ply('example1.ply', [coord, feat.astype(np.uint8), label], field_names)
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    elif repeat_align and label.shape[0] <= voxel_max:  # if M <= num_pts, directly repeat:
        coord, feat, label = data_dup(coord, feat, label, voxel_max)  # [num_point,cin]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)

    # print("Start check")
    if torch.isnan(coord).any() == torch.tensor(True) or torch.isnan(feat).any() == torch.tensor(True) or torch.isnan(label).any() == torch.tensor(True):
                print("!")
                exit(0)
    if torch.isinf(coord).any() == torch.tensor(True) or torch.isinf(feat).any() == torch.tensor(True) or torch.isinf(label).any() == torch.tensor(True):
                print("!")
                exit(0)  
    return coord, feat, label


def data_prepare_v101_scannet(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
    # # print(label.shape)
    keep_idx = np.where((label <= 19.))
    coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
    # print("!", label.shape)
    
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        # print("voxelized", label.shape)
        # # visualize
        # field_names = ['x', 'y', 'z', 'red', 'green', 'blue','values']
        # # write_ply('example1.ply', [coord, feat.astype(np.uint8), label], field_names)
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    elif repeat_align and label.shape[0] <= voxel_max:  # if M <= num_pts, directly repeat:
        coord, feat, label = data_dup(coord, feat, label, voxel_max)  # [num_point,cin]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)

    # print("Start check")
    if torch.isnan(coord).any() == torch.tensor(True) or torch.isnan(feat).any() == torch.tensor(True) or torch.isnan(label).any() == torch.tensor(True):
                print("!")
                exit(0)
    if torch.isinf(coord).any() == torch.tensor(True) or torch.isinf(feat).any() == torch.tensor(True) or torch.isinf(label).any() == torch.tensor(True):
                print("!")
                exit(0)  
    return coord, feat, label


def data_prepare_v101_desk_other(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
    # # print(label.shape)
    # keep_idx = np.where((label != 255.))
    # coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
    # print("!", label.shape)
    
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        # print("voxelized", label.shape)
        # # visualize
        # field_names = ['x', 'y', 'z', 'red', 'green', 'blue','values']
        # # write_ply('example1.ply', [coord, feat.astype(np.uint8), label], field_names)
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    elif repeat_align and label.shape[0] <= voxel_max:  # if M <= num_pts, directly repeat:
        coord, feat, label = data_dup(coord, feat, label, voxel_max)  # [num_point,cin]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    label[label==255]=256   # other furniture to 256
    label[label==53]=255  # artifacts as 255 to be ignored
    label[label==256]=53  # other furniture to 53

    # print("Start check")
    if torch.isnan(coord).any() == torch.tensor(True) or torch.isnan(feat).any() == torch.tensor(True) or torch.isnan(label).any() == torch.tensor(True):
                print("!")
                exit(0)
    if torch.isinf(coord).any() == torch.tensor(True) or torch.isinf(feat).any() == torch.tensor(True) or torch.isinf(label).any() == torch.tensor(True):
                print("!")
                exit(0)  
    return coord, feat, label


def data_prepare_v101_bin_desk_other_label(coord, feat, label, split='train', bin_num=2, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
    # # print(label.shape)
    # keep_idx = np.where((label != 255.))
    # coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
    # print("!", label.shape)
    
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        # print("voxelized", label.shape)
        # # visualize
        # field_names = ['x', 'y', 'z', 'red', 'green', 'blue','values']
        # # write_ply('example1.ply', [coord, feat.astype(np.uint8), label], field_names)
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    elif repeat_align and label.shape[0] <= voxel_max:  # if M <= num_pts, directly repeat:
        coord, feat, label = data_dup(coord, feat, label, voxel_max)  # [num_point,cin]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label_ori = torch.LongTensor(label)
    label_bin = torch.LongTensor(label)

    label_ori[label_ori==255]=256   # other furniture to 256
    label_ori[label_ori==53]=255  # artifacts as 255 to be ignored
    label_ori[label_ori==256]=53  # other furniture to 53
    
    if bin_num==2:
        label_bin[label_bin<=52]=1   # table obj to 1
        label_bin[label_bin!=1]=0  # others to 0
    elif bin_num==3:
        label_bin[label_bin<=52]=1   # table obj to 1
        label_bin[label_bin==53]=2   # artifacts to 2
        label_bin[label_bin==255]=0  # others to 0

    return coord, feat, label_ori, label_bin


def data_prepare_v101_bin(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
    # # print(label.shape)
    # keep_idx = np.where((label != 255.))
    # coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
    # print("!", label.shape)
    
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        # print("voxelized", label.shape)
        # # visualize
        # field_names = ['x', 'y', 'z', 'red', 'green', 'blue','values']
        # # write_ply('example1.ply', [coord, feat.astype(np.uint8), label], field_names)
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    elif repeat_align and label.shape[0] <= voxel_max:  # if M <= num_pts, directly repeat:
        coord, feat, label = data_dup(coord, feat, label, voxel_max)  # [num_point,cin]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label_ori = torch.LongTensor(label)
    label_bin = torch.LongTensor(label)

    label_bin[label_bin<=19]=0  # out of table, 255 still 255
    label_bin[label_bin==255]=6 # 255 to 6, which < 19
    label_bin[label_bin>19]=1  # on table, 6 still 6
    label_bin[label_bin==6]=255  # 6 to 255 again

    # print("Start check")
    if torch.isnan(coord).any() == torch.tensor(True) or torch.isnan(feat).any() == torch.tensor(True) or torch.isnan(label_ori).any() == torch.tensor(True):
                print("!")
                exit(0)
    if torch.isinf(coord).any() == torch.tensor(True) or torch.isinf(feat).any() == torch.tensor(True) or torch.isinf(label_ori).any() == torch.tensor(True):
                print("!")
                exit(0)  
                
    return coord, feat, label_ori, label_bin


def data_prepare_v101_bin_v2(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
    # # print(label.shape)
    # keep_idx = np.where((label != 255.))
    # coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
    # print("!", label.shape)
    
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        # print("voxelized", label.shape)
        # # visualize
        # field_names = ['x', 'y', 'z', 'red', 'green', 'blue','values']
        # # write_ply('example1.ply', [coord, feat.astype(np.uint8), label], field_names)
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    elif repeat_align and label.shape[0] <= voxel_max:  # if M <= num_pts, directly repeat:
        coord, feat, label = data_dup(coord, feat, label, voxel_max)  # [num_point,cin]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label_ori = torch.LongTensor(label)
    label_bin = torch.LongTensor(label)

    label_bin[label_bin<=19]=0  # out of table, 255 still 255
    label_bin[label_bin==255]=6 # 255 to 6, which < 19
    label_bin[label_bin>19]=1  # on table, 6 still 6
    label_bin[label_bin==6]=0  # 6 to 255 again

    # print("Start check")
    if torch.isnan(coord).any() == torch.tensor(True) or torch.isnan(feat).any() == torch.tensor(True) or torch.isnan(label_ori).any() == torch.tensor(True):
                print("!")
                exit(0)
    if torch.isinf(coord).any() == torch.tensor(True) or torch.isinf(feat).any() == torch.tensor(True) or torch.isinf(label_ori).any() == torch.tensor(True):
                print("!")
                exit(0)  
                
    return coord, feat, label_ori, label_bin


def data_prepare_v101_bin_label(coord, feat, label, split='train', bin_num=2, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
    # # print(label.shape)
    # keep_idx = np.where((label != 255.))
    # coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
    # print("!", label.shape)
    
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        # print("voxelized", label.shape)
        # # visualize
        # field_names = ['x', 'y', 'z', 'red', 'green', 'blue','values']
        # # write_ply('example1.ply', [coord, feat.astype(np.uint8), label], field_names)
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    elif repeat_align and label.shape[0] <= voxel_max:  # if M <= num_pts, directly repeat:
        coord, feat, label = data_dup(coord, feat, label, voxel_max)  # [num_point,cin]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label_ori = torch.LongTensor(label)
    label_bin = torch.LongTensor(label)

    label_bin[label_bin<=19]=0  # out of table, 255 still 255
    label_bin[label_bin==255]=6 # 255 to 6, which < 19
    label_bin[label_bin>19]=1  # on table, 6 still 6
    if bin_num == 2:
        label_bin[label_bin==6]=0  # 6 to 0
    elif bin_num == 3:
        label_bin[label_bin==6]=2  # 6 to 2
    else:
        raise Exception('bin_num not supported yet')
    
    # print("Start check")
    if torch.isnan(coord).any() == torch.tensor(True) or torch.isnan(feat).any() == torch.tensor(True) or torch.isnan(label_ori).any() == torch.tensor(True):
                print("!")
                exit(0)
    if torch.isinf(coord).any() == torch.tensor(True) or torch.isinf(feat).any() == torch.tensor(True) or torch.isinf(label_ori).any() == torch.tensor(True):
                print("!")
                exit(0)
                
    return coord, feat, label_ori, label_bin


def data_prepare_v101_bin_label_desk_other(coord, feat, label, label2, split='train', bin_num=2, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
    # # print(label.shape)
    # keep_idx = np.where((label != 255.))
    # coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
    # print("!", label.shape)

    if transform:
        coord, feat, label = transform(coord, feat, label)
        _, _, label2 = transform(coord, feat, label2)

    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label, label2 = coord[uniq_idx], feat[uniq_idx], label[uniq_idx], label2[uniq_idx]
        # print("voxelized", label.shape)
        # # visualize
        # field_names = ['x', 'y', 'z', 'red', 'green', 'blue','values']
        # # write_ply('example1.ply', [coord, feat.astype(np.uint8), label], field_names)
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label, label2 = coord[crop_idx], feat[crop_idx], label[crop_idx], label2[crop_idx]
    elif repeat_align and label.shape[0] <= voxel_max:  # if M <= num_pts, directly repeat:
        coord, feat, label = data_dup(coord, feat, label, voxel_max)  # [num_point,cin]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label, label2 = coord[shuf_idx], feat[shuf_idx], label[shuf_idx], label2[shuf_idx]
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label_ori = torch.LongTensor(label)
    label_bin = torch.LongTensor(label2)

    label_bin[label_bin<=19]=0  # out of table, 255 still 255
    label_bin[label_bin==255]=6 # 255 to 6, which < 19
    label_bin[label_bin>19]=1  # on table, 6 still 6
    if bin_num == 2:
        label_bin[label_bin==6]=0  # 6 to 0
    elif bin_num == 3:
        label_bin[label_bin==6]=2  # 6 to 2
    else:
        raise Exception('bin_num not supported yet')
                
    return coord, feat, label_ori, label_bin


def data_prepare_v101_bin_sig(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
    # # print(label.shape)
    # keep_idx = np.where((label != 255.))
    # coord, feat, label = coord[keep_idx], feat[keep_idx], label[keep_idx]
    # print("!", label.shape)
    
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        # print("voxelized", label.shape)
        # # visualize
        # field_names = ['x', 'y', 'z', 'red', 'green', 'blue','values']
        # # write_ply('example1.ply', [coord, feat.astype(np.uint8), label], field_names)
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    elif repeat_align and label.shape[0] <= voxel_max:  # if M <= num_pts, directly repeat:
        coord, feat, label = data_dup(coord, feat, label, voxel_max)  # [num_point,cin]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label_ori = torch.LongTensor(label)
    label_bin = torch.FloatTensor(label)
    
    label_bin[label_bin<=19.0]=0.  # original scannet_obj, 0
    label_bin[label_bin==255.]=0.  # ignored, 0
    label_bin[label_bin!=0.]=1.  # others means table_obj, 1

    # print("Start check")
    if torch.isnan(coord).any() == torch.tensor(True) or torch.isnan(feat).any() == torch.tensor(True) or torch.isnan(label_ori).any() == torch.tensor(True):
                print("!")
                exit(0)
    if torch.isinf(coord).any() == torch.tensor(True) or torch.isinf(feat).any() == torch.tensor(True) or torch.isinf(label_ori).any() == torch.tensor(True):
                print("!")
                exit(0)  
                
    return coord, feat, label_ori, label_bin


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= (coord_min + coord_max) / 2.0
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v102(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    while voxel_max and label.shape[0] > voxel_max * 1.1:
        area_rate = voxel_max / float(label.shape[0])
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        x_max, y_max = coord_max[0:2]
        x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
        if split == 'train':
            x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
        else:
            x_s, y_s = 0, 0
        x_e, y_e = x_s + x_size, y_s + y_size
        crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
        if crop_idx.shape[0] < voxel_max // 8: continue
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]

    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v103(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        xy_area = 7
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(xy_area)
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[1] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[1] * (y_area + 1) / float(xy_area)
            crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
            if crop_idx.shape[0] > 0:
                init_idx = crop_idx[np.random.randint(crop_idx.shape[0])] if 'train' in split else label.shape[0] // 2
                crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
                coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v104(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        xy_area = 10
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(xy_area)
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[1] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[1] * (y_area + 1) / float(xy_area)
            crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
            if crop_idx.shape[0] > 0:
                init_idx = crop_idx[np.random.randint(crop_idx.shape[0])] if 'train' in split else label.shape[0] // 2
                crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
                coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v105(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord[:, 0:2] -= coord_min[0:2]
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label
