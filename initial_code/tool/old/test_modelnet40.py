import os
import time
import numpy as np
import logging
import yaml
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from util import dataset, transform
from util.util import AverageMeter, intersectionAndUnion


random.seed(123)
np.random.seed(123)


def get_parser():
    parser = ArgumentParser(description='PyTorch PointNet Classification / Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/modelnet40/modelnet40_pointweb.yaml', help='config file')
    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)
    return args_cfg


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
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
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
    elif args.arch == 'pointtransformer_cls26_v11':
        from model.pointtransformer.pointtransformer_cls_v11 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v12':
        from model.pointtransformer.pointtransformer_cls_v12 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls38_v11':
        from model.pointtransformer.pointtransformer_cls_v11 import pointtransformer_cls38 as Model
    elif args.arch == 'pointtransformer_cls38_v12':
        from model.pointtransformer.pointtransformer_cls_v12 import pointtransformer_cls38 as Model
    elif args.arch == 'pointtransformer_cls26_v13':
        from model.pointtransformer.pointtransformer_cls_v13 import pointtransformer_cls26 as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    # model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz)
    model = Model(c=args.fea_dim, k=args.classes)
    model = torch.nn.DataParallel(model.cuda())
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.PointData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform, num_point=args.num_point)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(test_loader, model, criterion, names)


def rotate(xyz_normal, angle):
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    matrix = torch.tensor([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]).cuda()
    xyz_normal_rotate = torch.zeros_like(xyz_normal)
    xyz_normal_rotate[:, :, 0:3] = torch.matmul(xyz_normal[:, :, 0:3], matrix)
    xyz_normal_rotate[:, :, 3:6] = torch.matmul(xyz_normal[:, :, 3:6], matrix)
    return xyz_normal_rotate

 
def test(test_loader, model, criterion, names, num_repeat=5):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        target = target[:, 0]  # for cls
        output_all = np.zeros((input.shape[0], args.classes))
        loss = torch.tensor(0).float().cuda()
        for j in range(num_repeat):
            #input_tmp = rotate(input, j / float(num_repeat) * np.pi * 2)
            input_tmp = torch.zeros_like(input)
            for k in range(input.shape[0]):
                idx = np.random.randint(input.shape[1], size=input.shape[1])
                input_tmp[k] = input[k, idx, :]
            with torch.no_grad():
                output = model(input_tmp)
            loss += criterion(output, target)
            output_all += output.data.cpu().numpy()
            #output = output.data.max(1)[1].cpu().numpy()
            #for k in range(output.size):
            #    output_all[k, output[k]] += 1
        output = np.argmax(output_all, 1)
        loss /= num_repeat
        target = target.cpu().numpy()
        intersection, union, target = intersectionAndUnion(output, target, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % args.print_freq == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(test_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()
