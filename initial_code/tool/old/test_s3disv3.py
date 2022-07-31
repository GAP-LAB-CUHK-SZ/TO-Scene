import os
import time
import random
import numpy as np
import logging
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs
from util.s3disv3 import grid_sub_sampling

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
    elif args.arch == 'pointnet2_seg':
        from model.pointnet2.pointnet2_seg import PointNet2SSGSeg as Model
    elif args.arch == 'pointweb_seg':
        from model.pointweb.pointweb_seg import PointWebSeg as Model
    elif args.arch == 'pointnet2_cls':
        from model.pointnet2.pointnet2_cls import PointNet2SSGCls as Model
    elif args.arch == 'pointnet2_cls_v2':
        from model.pointnet2.pointnet2_cls_v2 import PointNet2SSGCls as Model
    elif args.arch == 'pointtransformer_cls26_v1':
        from model.pointtransformer.pointtransformer_cls_v1 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v2':
        from model.pointtransformer.pointtransformer_cls_v2 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v3':
        from model.pointtransformer.pointtransformer_cls_v3 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v4':
        from model.pointtransformer.pointtransformer_cls_v4 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v5':
        from model.pointtransformer.pointtransformer_cls_v5 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v6':
        from model.pointtransformer.pointtransformer_cls_v6 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v7':
        from model.pointtransformer.pointtransformer_cls_v7 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v8':
        from model.pointtransformer.pointtransformer_cls_v8 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v9':
        from model.pointtransformer.pointtransformer_cls_v9 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v10':
        from model.pointtransformer.pointtransformer_cls_v10 import pointtransformer_cls26 as Model
    elif args.arch == 'pointweb_seg_v2':
        from model.pointweb.pointweb_seg_v2 import PointWebSeg as Model
    
    elif args.arch == 'pointtransformer_seg26_v1':
        from model.pointtransformer.pointtransformer_seg_v1 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v2':
        from model.pointtransformer.pointtransformer_seg_v2 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v3':
        from model.pointtransformer.pointtransformer_seg_v3 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v4':
        from model.pointtransformer.pointtransformer_seg_v4 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v5':
        from model.pointtransformer.pointtransformer_seg_v5 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v6':
        from model.pointtransformer.pointtransformer_seg_v6 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v6':
        from model.pointtransformer.pointtransformer_seg_v6 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v7':
        from model.pointtransformer.pointtransformer_seg_v7 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v8':
        from model.pointtransformer.pointtransformer_seg_v8 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v9':
        from model.pointtransformer.pointtransformer_seg_v9 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v10':
        from model.pointtransformer.pointtransformer_seg_v10 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v11':
        from model.pointtransformer.pointtransformer_seg_v11 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v12':
        from model.pointtransformer.pointtransformer_seg_v12 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v13':
        from model.pointtransformer.pointtransformer_seg_v13 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v14':
        from model.pointtransformer.pointtransformer_seg_v14 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v15':
        from model.pointtransformer.pointtransformer_seg_v15 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v16':
        from model.pointtransformer.pointtransformer_seg_v16 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v17':
        from model.pointtransformer.pointtransformer_seg_v17 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v18':
        from model.pointtransformer.pointtransformer_seg_v18 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v19':
        from model.pointtransformer.pointtransformer_seg_v19 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v20':
        from model.pointtransformer.pointtransformer_seg_v20 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v21':
        from model.pointtransformer.pointtransformer_seg_v21 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v22':
        from model.pointtransformer.pointtransformer_seg_v22 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v23':
        from model.pointtransformer.pointtransformer_seg_v23 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v34':
        from model.pointtransformer.pointtransformer_seg_v34 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v35':
        from model.pointtransformer.pointtransformer_seg_v35 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v36':
        from model.pointtransformer.pointtransformer_seg_v36 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v37':
        from model.pointtransformer.pointtransformer_seg_v37 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v37':
        from model.pointtransformer.pointtransformer_seg_v37 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v38':
        from model.pointtransformer.pointtransformer_seg_v38 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v39':
        from model.pointtransformer.pointtransformer_seg_v39 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v40':
        from model.pointtransformer.pointtransformer_seg_v40 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v40':
        from model.pointtransformer.pointtransformer_seg_v40 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v41':
        from model.pointtransformer.pointtransformer_seg_v41 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v41':
        from model.pointtransformer.pointtransformer_seg_v41 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v42':
        from model.pointtransformer.pointtransformer_seg_v42 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v43':
        from model.pointtransformer.pointtransformer_seg_v43 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v44':
        from model.pointtransformer.pointtransformer_seg_v44 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v45':
        from model.pointtransformer.pointtransformer_seg_v45 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v46':
        from model.pointtransformer.pointtransformer_seg_v46 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v47':
        from model.pointtransformer.pointtransformer_seg_v47 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v48':
        from model.pointtransformer.pointtransformer_seg_v48 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v49':
        from model.pointtransformer.pointtransformer_seg_v49 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v50':
        from model.pointtransformer.pointtransformer_seg_v50 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v51':
        from model.pointtransformer.pointtransformer_seg_v51 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v52':
        from model.pointtransformer.pointtransformer_seg_v52 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v53':
        from model.pointtransformer.pointtransformer_seg_v53 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v54':
        from model.pointtransformer.pointtransformer_seg_v54 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v55':
        from model.pointtransformer.pointtransformer_seg_v55 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v56':
        from model.pointtransformer.pointtransformer_seg_v56 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v57':
        from model.pointtransformer.pointtransformer_seg_v57 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v58':
        from model.pointtransformer.pointtransformer_seg_v58 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v59':
        from model.pointtransformer.pointtransformer_seg_v59 import pointtransformer_seg26 as Model
        
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    # model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz)
    model = Model(c=args.fea_dim, k=args.classes)
    model = torch.nn.DataParallel(model.cuda())
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion, names)


def data_prepare(room_path):
    room_data = np.load(room_path)
    points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
    # coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

    xyz = room_data[:, :3].astype(np.float32)
    colors = room_data[:, 3:6].astype(np.uint8)
    labels = room_data[:, 6].astype(np.uint8)
    sub_xyz, sub_colors, sub_labels = grid_sub_sampling(xyz, colors, labels, args.block_size)
    coord_min, coord_max = np.amin(sub_xyz, axis=0)[:3], np.amax(sub_xyz, axis=0)[:3]
    sub_xyz -= (coord_max - coord_min) / 2
    sub_colors = sub_colors / 255.0

    point_idxs = range(sub_xyz.shape[0])
    selected_idxs = np.random.choice(point_idxs, args.num_point, replace=False if sub_xyz.shape[0] >= args.num_point else True)
    selected_xyz = sub_xyz[selected_idxs, :]
    selected_colors = sub_colors[selected_idxs, :]
    selected_labels = np.squeeze(sub_labels[selected_idxs], axis=-1)
    selected_points = np.concatenate((selected_xyz, selected_colors), 1)
    return selected_points, selected_labels


def test(model, criterion, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    rooms = sorted(os.listdir(args.train_full_folder))
    rooms = [room for room in rooms if 'Area_' in room]
    rooms_split = [room for room in rooms if 'Area_{}'.format(args.test_area) in room]
    gt_all, pred_all = np.array([]), np.array([])
    check_makedirs(args.save_folder)
    pred_save, gt_save = [], []
    for idx, room_name in enumerate(rooms_split):
        input, target = data_prepare(os.path.join(args.train_full_folder, room_name))
        end = time.time()
        input = torch.from_numpy(input).float().view(-1, args.num_point, input.shape[1])
        target = torch.from_numpy(target).long().view(-1, args.num_point)
        with torch.no_grad():
            output = model(input.cuda())
        loss = criterion(output, target.cuda())  # for reference
        output = output.max(1)[1]

        output, target = output.cpu().numpy(), target.cpu().numpy()
        intersection, union, target_tmp = intersectionAndUnion(output, target, args.classes, args.ignore_label)
        accuracy = sum(intersection) / (sum(target_tmp) + 1e-10)

        batch_time.update(time.time() - end)
        end = time.time()
        logger.info('Test: [{}/{}] '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss {loss:.4f} '
                    'Accuracy {accuracy:.4f} '
                    'Points {target.size}.'.format(idx + 1, len(rooms_split),
                                               batch_time=batch_time,
                                               loss=loss,
                                               accuracy=accuracy,
                                               target=target))
        # calculation 1: add per room predictions
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target_tmp)
        # calculation 2
        pred_all = np.hstack([pred_all, output]) if pred_all.size else output
        gt_all = np.hstack([gt_all, target]) if gt_all.size else target
        pred_save.append(output), gt_save.append(target)

    with open(os.path.join(args.save_folder, "pred_{}.pickle".format(args.test_area)), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "gt_{}.pickle".format(args.test_area)), 'wb') as handle:
        pickle.dump({'gt': gt_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = intersectionAndUnion(pred_all, gt_all, args.classes, args.ignore_label)
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
    return mIoU, mAcc, allAcc, pred_all


if __name__ == '__main__':
    main()
