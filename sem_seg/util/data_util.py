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


def data_prepare(coord, feat, label, split='train', voxel_size=0.004, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
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
    return coord, feat, label


def data_prepare_bin(coord, feat, label, split='train', voxel_size=0.004, voxel_max=None, transform=None, shuffle_index=False, repeat_align=False):
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
                
    return coord, feat, label_ori, label_bin