import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import dataset, transform, config
#from util.s3dis import S3DIS
#from util.s3disv3 import S3DIS
#from util.s3disv4_16 import S3DIS
#from util.s3disv4_xyflip import S3DIS
from util.s3disv4 import S3DIS
from util.scannet import ScanNet
from util.scannetv2 import ScanNetV2
from util.commom_util import AverageMeter, intersectionAndUnionGPU, find_free_port, poly_learning_rate
from util.transform import RandomScale, RandomRotate, RandomShift, RandomFlip, RandomJitter, RandomColor


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
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


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manualSeed)
        torch.cuda.manual_seed(args.manualSeed)
        torch.cuda.manual_seed_all(args.manualSeed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name == 's3dis':
        S3DIS(split='train', data_root=args.train_full_folder, num_point=args.num_point, test_area=args.test_area, sample_density=args.sample_density, shape_rate=args.shape_rate, sample_rate=args.sample_rate, transform=None)
    elif args.data_name == 'scannetv2':
        ScanNetV2(split='train', data_root=args.train_full_folder, num_point=args.num_point, sample_density=args.sample_density, shape_rate=args.shape_rate, sample_rate=args.sample_rate, transform=None)

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

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
    elif args.arch == 'pointnet2_cls_v3':
        from model.pointnet2.pointnet2_cls_v3 import PointNet2SSGCls as Model
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
    elif args.arch == 'pointtransformer_cls26_v14':
        from model.pointtransformer.pointtransformer_cls_v14 import pointtransformer_cls26 as Model
        model = Model(c=args.fea_dim, k=args.classes, p=args.droprate)
    elif args.arch == 'pointtransformer_cls26_v15':
        from model.pointtransformer.pointtransformer_cls_v15 import pointtransformer_cls26 as Model
    elif args.arch == 'pointtransformer_cls26_v16':
        from model.pointtransformer.pointtransformer_cls_v16 import pointtransformer_cls26 as Model
        model = Model(c=args.fea_dim, k=args.classes, p=args.droprate)
    elif args.arch == 'pointtransformer_cls26_v17':
        from model.pointtransformer.pointtransformer_cls_v17 import pointtransformer_cls26 as Model
        model = Model(c=args.fea_dim, k=args.classes)
    elif args.arch == 'pointtransformer_cls26_v18':
        from model.pointtransformer.pointtransformer_cls_v18 import pointtransformer_cls26 as Model
        model = Model(c=args.fea_dim, k=args.classes)
    elif args.arch == 'pointtransformer_cls26_v19':
        from model.pointtransformer.pointtransformer_cls_v19 import pointtransformer_cls26 as Model
        model = Model(c=args.fea_dim, k=args.classes)
    elif args.arch == 'pointtransformer_cls26_v20':
        from model.pointtransformer.pointtransformer_cls_v20 import pointtransformer_cls26 as Model
        model = Model(c=args.fea_dim, k=args.classes)

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
    elif args.arch == 'pointtransformer_seg26_v24':
        from model.pointtransformer.pointtransformer_seg_v24 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v25':
        from model.pointtransformer.pointtransformer_seg_v25 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v26':
        from model.pointtransformer.pointtransformer_seg_v26 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v27':
        from model.pointtransformer.pointtransformer_seg_v27 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v28':
        from model.pointtransformer.pointtransformer_seg_v28 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v28':
        from model.pointtransformer.pointtransformer_seg_v28 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg50_v28':
        from model.pointtransformer.pointtransformer_seg_v28 import pointtransformer_seg50 as Model
    elif args.arch == 'pointtransformer_seg26_v29':
        from model.pointtransformer.pointtransformer_seg_v29 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v30':
        from model.pointtransformer.pointtransformer_seg_v30 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v31':
        from model.pointtransformer.pointtransformer_seg_v31 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v32':
        from model.pointtransformer.pointtransformer_seg_v32 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v33':
        from model.pointtransformer.pointtransformer_seg_v33 import pointtransformer_seg26 as Model
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
    elif args.arch == 'pointtransformer_seg50_v37':
        from model.pointtransformer.pointtransformer_seg_v37 import pointtransformer_seg50 as Model
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
    elif args.arch == 'pointtransformer_seg26_v60':
        from model.pointtransformer.pointtransformer_seg_v60 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v61':
        from model.pointtransformer.pointtransformer_seg_v61 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v62':
        from model.pointtransformer.pointtransformer_seg_v62 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v63':
        from model.pointtransformer.pointtransformer_seg_v63 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v64':
        from model.pointtransformer.pointtransformer_seg_v64 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v65':
        from model.pointtransformer.pointtransformer_seg_v65 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v66':
        from model.pointtransformer.pointtransformer_seg_v66 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v66':
        from model.pointtransformer.pointtransformer_seg_v66 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg50_v66':
        from model.pointtransformer.pointtransformer_seg_v66 import pointtransformer_seg50 as Model
    elif args.arch == 'pointtransformer_seg26_v67':
        from model.pointtransformer.pointtransformer_seg_v67 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v68':
        from model.pointtransformer.pointtransformer_seg_v68 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v68':
        from model.pointtransformer.pointtransformer_seg_v68 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg50_v68':
        from model.pointtransformer.pointtransformer_seg_v68 import pointtransformer_seg50 as Model
    elif args.arch == 'pointtransformer_seg26_v69':
        from model.pointtransformer.pointtransformer_seg_v69 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v70':
        from model.pointtransformer.pointtransformer_seg_v70 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v70':
        from model.pointtransformer.pointtransformer_seg_v70 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v71':
        from model.pointtransformer.pointtransformer_seg_v71 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v72':
        from model.pointtransformer.pointtransformer_seg_v72 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v72':
        from model.pointtransformer.pointtransformer_seg_v72 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v73':
        from model.pointtransformer.pointtransformer_seg_v73 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v74':
        from model.pointtransformer.pointtransformer_seg_v74 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v75':
        from model.pointtransformer.pointtransformer_seg_v75 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v76':
        from model.pointtransformer.pointtransformer_seg_v76 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v77':
        from model.pointtransformer.pointtransformer_seg_v77 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v78':
        from model.pointtransformer.pointtransformer_seg_v78 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v79':
        from model.pointtransformer.pointtransformer_seg_v79 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v80':
        from model.pointtransformer_v1.pointtransformer_seg_v80 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v80':
        from model.pointtransformer.pointtransformer_seg_v80 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg26_v81':
        from model.pointtransformer.pointtransformer_seg_v81 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v82':
        from model.pointtransformer.pointtransformer_seg_v82 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v83':
        from model.pointtransformer.pointtransformer_seg_v83 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v84':
        from model.pointtransformer.pointtransformer_seg_v84 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v85':
        from model.pointtransformer.pointtransformer_seg_v85 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v86':
        from model.pointtransformer.pointtransformer_seg_v86 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v87':
        from model.pointtransformer.pointtransformer_seg_v87 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v88':
        from model.pointtransformer.pointtransformer_seg_v88 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v89':
        from model.pointtransformer.pointtransformer_seg_v89 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v89':
        from model.pointtransformer.pointtransformer_seg_v89 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg50_v89':
        from model.pointtransformer.pointtransformer_seg_v89 import pointtransformer_seg50 as Model
    elif args.arch == 'pointtransformer_seg26_v90':
        from model.pointtransformer.pointtransformer_seg_v90 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v91':
        from model.pointtransformer.pointtransformer_seg_v91 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v91':
        from model.pointtransformer.pointtransformer_seg_v91 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg50_v91':
        from model.pointtransformer.pointtransformer_seg_v91 import pointtransformer_seg50 as Model
    elif args.arch == 'pointtransformer_seg26_v92':
        from model.pointtransformer.pointtransformer_seg_v92 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v93':
        from model.pointtransformer.pointtransformer_seg_v93 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v94':
        from model.pointtransformer.pointtransformer_seg_v94 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v95':
        from model.pointtransformer.pointtransformer_seg_v95 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v96':
        from model.pointtransformer.pointtransformer_seg_v96 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v97':
        from model.pointtransformer.pointtransformer_seg_v97 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v98':
        from model.pointtransformer.pointtransformer_seg_v98 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v99':
        from model.pointtransformer.pointtransformer_seg_v99 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v100':
        from model.pointtransformer.pointtransformer_seg_v100 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v101':
        from model.pointtransformer.pointtransformer_seg_v101 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v102':
        from model.pointtransformer.pointtransformer_seg_v102 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v103':
        from model.pointtransformer.pointtransformer_seg_v103 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v104':
        from model.pointtransformer.pointtransformer_seg_v104 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v105':
        from model.pointtransformer.pointtransformer_seg_v105 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v106':
        from model.pointtransformer.pointtransformer_seg_v106 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v107':
        from model.pointtransformer.pointtransformer_seg_v107 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v108':
        from model.pointtransformer.pointtransformer_seg_v108 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v109':
        from model.pointtransformer.pointtransformer_seg_v109 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v110':
        from model.pointtransformer.pointtransformer_seg_v110 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v111':
        from model.pointtransformer.pointtransformer_seg_v111 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v112':
        from model.pointtransformer.pointtransformer_seg_v112 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg38_v112':
        from model.pointtransformer.pointtransformer_seg_v112 import pointtransformer_seg38 as Model
    elif args.arch == 'pointtransformer_seg50_v112':
        from model.pointtransformer.pointtransformer_seg_v112 import pointtransformer_seg50 as Model
    elif args.arch == 'pointtransformer_seg26_v113':
        from model.pointtransformer.pointtransformer_seg_v113 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v114':
        from model.pointtransformer.pointtransformer_seg_v114 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v115':
        from model.pointtransformer.pointtransformer_seg_v115 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v116':
        from model.pointtransformer.pointtransformer_seg_v116 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v117':
        from model.pointtransformer.pointtransformer_seg_v117 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v118':
        from model.pointtransformer.pointtransformer_seg_v118 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v119':
        from model.pointtransformer.pointtransformer_seg_v119 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v120':
        from model.pointtransformer.pointtransformer_seg_v120 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v121':
        from model.pointtransformer.pointtransformer_seg_v121 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v122':
        from model.pointtransformer.pointtransformer_seg_v122 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v123':
        from model.pointtransformer.pointtransformer_seg_v123 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v124':
        from model.pointtransformer.pointtransformer_seg_v124 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v125':
        from model.pointtransformer.pointtransformer_seg_v125 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v126':
        from model.pointtransformer.pointtransformer_seg_v126 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v127':
        from model.pointtransformer.pointtransformer_seg_v127 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v128':
        from model.pointtransformer.pointtransformer_seg_v128 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v129':
        from model.pointtransformer.pointtransformer_seg_v129 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v130':
        from model.pointtransformer.pointtransformer_seg_v130 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v131':
        from model.pointtransformer.pointtransformer_seg_v131 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v132':
        from model.pointtransformer.pointtransformer_seg_v132 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v133':
        from model.pointtransformer.pointtransformer_seg_v133 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v134':
        from model.pointtransformer.pointtransformer_seg_v134 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v135':
        from model.pointtransformer.pointtransformer_seg_v135 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v136':
        from model.pointtransformer.pointtransformer_seg_v136 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v137':
        from model.pointtransformer.pointtransformer_seg_v137 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v138':
        from model.pointtransformer.pointtransformer_seg_v138 import pointtransformer_seg26 as Model
    elif args.arch == 'pointtransformer_seg26_v139':
        from model.pointtransformer.pointtransformer_seg_v139 import pointtransformer_seg26 as Model

    elif args.arch == 'pointweb_seg_v2':
        from model.pointweb.pointweb_seg_v2 import PointWebSeg as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes)
    # model = Model(c=args.fea_dim, k=args.classes, p=args.drop_rate)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_epoch, gamma=args.multiplier)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            #best_iou = 40.0
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_transform = transform.Compose([transform.ToTensor()])
    # train_transform = transform.Compose([RandomScale(0.8, 1.2), transform.ToTensor()])
    # train_transform = transform.Compose([RandomShift(0.2), transform.ToTensor()])
    # train_transform = transform.Compose([RandomRotate(), transform.ToTensor()])
    # train_transform = transform.Compose([RandomFlip(), transform.ToTensor()])
    # train_transform = transform.Compose([RandomColor(), transform.ToTensor()])
    # train_transform = transform.Compose([RandomScale(), RandomShift(), RandomRotate(), RandomFlip(), transform.ToTensor()])
    if args.data_name == 's3dis':
        train_data = S3DIS(split='train', data_root=args.train_full_folder, num_point=args.num_point, test_area=args.test_area, sample_density=args.sample_density, shape_rate=args.shape_rate, sample_rate=args.sample_rate, transform=train_transform)
        if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
        # train_data = dataset.PointData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    elif args.data_name == 'scannet':
        train_data = ScanNet(split='train', data_root=args.data_root, num_point=args.num_point, block_size=args.block_size, sample_rate=args.sample_rate, transform=train_transform)

    elif args.data_name == 'scannetv2':
        train_data = ScanNetV2(split='train', data_root=args.train_full_folder, num_point=args.num_point, sample_density=args.sample_density, shape_rate=args.shape_rate, sample_rate=args.sample_rate, transform=train_transform)
        if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))

    elif args.data_name == 'modelnet40':
        train_data = dataset.PointData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform, num_point=args.num_point, random_index=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = None
    if args.evaluate:
        val_transform = transform.Compose([transform.ToTensor()])
        val_data = dataset.PointData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform, num_point=args.num_point)
        # val_data = S3DIS(split='val', data_root=args.train_full_folder, num_point=args.num_point, test_area=args.test_area, block_size=args.block_size, sample_rate=args.sample_rate, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        n = input.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        # poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=0.9)

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        output = output.max(1)[1]
        n = input.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()
