import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config
from util.common_util import AverageMeter, intersectionAndUnionGPU, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification / Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointweb.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'pointnet_seg':
        from model.pointnet.pointnet import PointNetSeg as Model
    elif args.arch == 'pointtransformer_seg26_v199':
        from model.pointtransformer.pointtransformer_seg_v199 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v200':
        from model.pointtransformer.pointtransformer_seg_v200 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v201':
        from model.pointtransformer.pointtransformer_seg_v201 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v202':
        from model.pointtransformer.pointtransformer_seg_v202 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v202':
        from model.pointtransformer.pointtransformer_seg_v202 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v203':
        from model.pointtransformer.pointtransformer_seg_v203 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v203':
        from model.pointtransformer.pointtransformer_seg_v203 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v204':
        from model.pointtransformer.pointtransformer_seg_v204 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v205':
        from model.pointtransformer.pointtransformer_seg_v205 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v206':
        from model.pointtransformer.pointtransformer_seg_v206 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v207':
        from model.pointtransformer.pointtransformer_seg_v207 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v208':
        from model.pointtransformer.pointtransformer_seg_v208 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v209':
        from model.pointtransformer.pointtransformer_seg_v209 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v210':
        from model.pointtransformer.pointtransformer_seg_v210 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v211':
        from model.pointtransformer.pointtransformer_seg_v211 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v212':
        from model.pointtransformer.pointtransformer_seg_v212 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v212':
        from model.pointtransformer.pointtransformer_seg_v212 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v213':
        from model.pointtransformer.pointtransformer_seg_v213 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v213':
        from model.pointtransformer.pointtransformer_seg_v213 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v214':
        from model.pointtransformer.pointtransformer_seg_v214 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v215':
        from model.pointtransformer.pointtransformer_seg_v215 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v216':
        from model.pointtransformer.pointtransformer_seg_v216 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v217':
        from model.pointtransformer.pointtransformer_seg_v217 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v218':
        from model.pointtransformer.pointtransformer_seg_v218 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v219':
        from model.pointtransformer.pointtransformer_seg_v219 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v220':
        from model.pointtransformer.pointtransformer_seg_v220 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v221':
        from model.pointtransformer.pointtransformer_seg_v221 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v222':
        from model.pointtransformer.pointtransformer_seg_v222 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v222':
        from model.pointtransformer.pointtransformer_seg_v222 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v223':
        from model.pointtransformer.pointtransformer_seg_v223 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v223':
        from model.pointtransformer.pointtransformer_seg_v223 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v224':
        from model.pointtransformer.pointtransformer_seg_v224 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v225':
        from model.pointtransformer.pointtransformer_seg_v225 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v226':
        from model.pointtransformer.pointtransformer_seg_v226 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v227':
        from model.pointtransformer.pointtransformer_seg_v227 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v228':
        from model.pointtransformer.pointtransformer_seg_v228 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v229':
        from model.pointtransformer.pointtransformer_seg_v229 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v230':
        from model.pointtransformer.pointtransformer_seg_v230 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v231':
        from model.pointtransformer.pointtransformer_seg_v231 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v232':
        from model.pointtransformer.pointtransformer_seg_v232 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v232':
        from model.pointtransformer.pointtransformer_seg_v232 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v233':
        from model.pointtransformer.pointtransformer_seg_v233 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v233':
        from model.pointtransformer.pointtransformer_seg_v233 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v234':
        from model.pointtransformer.pointtransformer_seg_v234 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v235':
        from model.pointtransformer.pointtransformer_seg_v235 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v236':
        from model.pointtransformer.pointtransformer_seg_v236 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v237':
        from model.pointtransformer.pointtransformer_seg_v237 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v238':
        from model.pointtransformer.pointtransformer_seg_v238 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v239':
        from model.pointtransformer.pointtransformer_seg_v239 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v230':
        from model.pointtransformer.pointtransformer_seg_v230 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v231':
        from model.pointtransformer.pointtransformer_seg_v231 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v232':
        from model.pointtransformer.pointtransformer_seg_v232 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v232':
        from model.pointtransformer.pointtransformer_seg_v232 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v233':
        from model.pointtransformer.pointtransformer_seg_v233 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v233':
        from model.pointtransformer.pointtransformer_seg_v233 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v234':
        from model.pointtransformer.pointtransformer_seg_v234 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v235':
        from model.pointtransformer.pointtransformer_seg_v235 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v236':
        from model.pointtransformer.pointtransformer_seg_v236 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v237':
        from model.pointtransformer.pointtransformer_seg_v237 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v238':
        from model.pointtransformer.pointtransformer_seg_v238 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v239':
        from model.pointtransformer.pointtransformer_seg_v239 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v240':
        from model.pointtransformer.pointtransformer_seg_v240 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v241':
        from model.pointtransformer.pointtransformer_seg_v241 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v242':
        from model.pointtransformer.pointtransformer_seg_v242 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v243':
        from model.pointtransformer.pointtransformer_seg_v243 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v244':
        from model.pointtransformer.pointtransformer_seg_v244 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v245':
        from model.pointtransformer.pointtransformer_seg_v245 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v246':
        from model.pointtransformer.pointtransformer_seg_v246 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v247':
        from model.pointtransformer.pointtransformer_seg_v247 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v248':
        from model.pointtransformer.pointtransformer_seg_v248 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v249':
        from model.pointtransformer.pointtransformer_seg_v249 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v250':
        from model.pointtransformer.pointtransformer_seg_v250 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v251':
        from model.pointtransformer.pointtransformer_seg_v251 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v252':
        from model.pointtransformer.pointtransformer_seg_v252 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v253':
        from model.pointtransformer.pointtransformer_seg_v253 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v254':
        from model.pointtransformer.pointtransformer_seg_v254 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v255':
        from model.pointtransformer.pointtransformer_seg_v255 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v256':
        from model.pointtransformer.pointtransformer_seg_v256 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v257':
        from model.pointtransformer.pointtransformer_seg_v257 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v258':
        from model.pointtransformer.pointtransformer_seg_v258 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v259':
        from model.pointtransformer.pointtransformer_seg_v259 import pointtransformer_seg26 as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes).cuda()
    #model = torch.nn.DataParallel(model.cuda())
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion, names)


def estimate_normal(coord, radius=0.1, max_nn=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normal = np.asarray(pcd.normals)
    return normal


def data_prepare(room_path):
    room_data = np.load(room_path)
    coord, feat, label = room_data[:, 0:3], room_data[:, 3:6], room_data[:, 6]
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= coord_min

    idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
    idx_list = []
    for i in range(count.max()):
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
        idx_part = idx_sort[idx_select]
        idx_list.append(idx_part)
    return coord, feat, label, idx_list


def test(model, criterion, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    args.batch_size_test=1
    model.eval()
    rooms = sorted(os.listdir(args.train_full_folder))
    rooms_split = [room for room in rooms if 'Area_{}'.format(args.test_area) in room]
    check_makedirs(args.save_folder)
    pred_save, gt_save = [], []
    for idx, room_name in enumerate(rooms_split):
        end = time.time()
        coord, feat, gt, idx_room = data_prepare(os.path.join(args.train_full_folder, room_name))
        pred = torch.zeros((gt.size, args.classes)).cuda()
        batch_num = int(np.ceil(len(idx_room) / args.batch_size_test))
        for i in range(batch_num):
            s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_room))
            idx_list, coord_list, feat_list, offset,  = [], [], [], []
            for j in range(s_i, e_i):
                idx_part = idx_room[j]
                coord_part, feat_part = coord[idx_part], feat[idx_part]
                feat_part = np.concatenate((feat_part, estimate_normal(coord_part)), 1)
                coord_min, coord_max = np.min(coord_part, 0), np.max(coord_part, 0)
                coord_part[:, 0:2] -= (coord_max[0:2] - coord_min[0:2]) / 2.0
                feat_part[:, :3] = feat_part[:, :3] / 127.5 - 1.
                idx_list.append(idx_part)
                coord_list.append(coord_part), feat_list.append(feat_part), offset.append(idx_part.size)
            idx_part = np.concatenate(idx_list)
            coord_part = torch.tensor(np.concatenate(coord_list)).float().cuda(non_blocking=True)
            feat_part = torch.tensor(np.concatenate(feat_list)).float().cuda(non_blocking=True)
            offset = torch.IntTensor(np.cumsum(offset)).cuda(non_blocking=True)
            with torch.no_grad():
                pred_part = model([coord_part, feat_part, offset])  # (n, k)
            torch.cuda.empty_cache()
            pred[idx_part, :] += pred_part
        loss = criterion(pred, torch.LongTensor(gt).cuda(non_blocking=True))  # for reference
        pred = pred.max(1)[1].data.cpu().numpy()

        # calculation 1: add per room predictions
        intersection, union, target = intersectionAndUnion(pred, gt, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        batch_time.update(time.time() - end)
        end = time.time()
        logger.info('Test: [{}/{}]-[{}/{}/{}] '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Accuracy {accuracy:.4f}.'.format(idx + 1, len(rooms_split), batch_num, len(idx_room) * idx_room[0].size, gt.size,
                                               batch_time=batch_time, accuracy=accuracy))
        pred_save.append(pred), gt_save.append(gt)

    #with open(os.path.join(args.save_folder, "pred_{}.pickle".format(args.test_area)), 'wb') as handle:
    #    pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(os.path.join(args.save_folder, "gt_{}.pickle".format(args.test_area)), 'wb') as handle:
    #    pickle.dump({'gt': gt_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(gt_save), args.classes, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
