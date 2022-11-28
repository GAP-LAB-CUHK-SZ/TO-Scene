import torch
from tensorboardX import SummaryWriter
import os
import datetime
import time
import numpy as np
import pickle
import torch.nn as nn
from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from data.model_utils_DOS import DOS_scene_config,DOS_desk_config

def detect_trainer(cfg,model,optimizer,scheduler,train_loader,test_loader,checkpoint):
    start_t = time.time()
    config = cfg.config
    if config['data']['dataset']=="TOS_desk":
        DC=DOS_desk_config()
    elif config['data']['dataset']=="TOS_scene":
        DC=DOS_scene_config()
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)
    start_epoch = 0
    if config["resume"] == True:
        checkpoint.load(config["weight"])
        start_epoch = scheduler.last_epoch
    scheduler.last_epoch = start_epoch

    if config['optimizer']['use_bnm_scheduler']:
        from external.pointnet2.pytorch_utils import BNMomentumScheduler
        BN_DECAY_RATE=config['optimizer']['bn_decay_rate']
        BN_DECAY_STEP=config['optimizer']['bn_decay_step']
        BN_MOMENTUM_INIT = 0.5
        BN_MOMENTUM_MAX = 0.001
        bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
        bnm_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch)

    #-----------------------evaluate config---------------------------------------
    CONFIG_DICT = {'remove_empty_box': True, 'use_3d_nms': True,
                   'nms_iou': 0.5, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.05,
                   'dataset_config': DC}

    ap_iou_thresh=config['data']['ap_iou_thresh']

    model.train()
    iter = 0
    min_eval_loss = 10000
    for e in range(start_epoch, config['other']['nepoch']):
        cfg.log_string("Switch Phase to Train")
        if config['optimizer']['use_bnm_scheduler']:
            bnm_scheduler.step()
        model.train()
        for batch_id, data_batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            for key in data_batch:
                if isinstance(data_batch[key], list) == False:
                    data_batch[key] = data_batch[key].cuda()
            est_data, loss = model(data_batch)
            total_loss = torch.mean(loss)
            total_loss.backward()
            optimizer.step()
            if iter % config['other']['log_interval']==0:
                msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    "epoch",
                    e,
                    batch_id + 1,
                    len(train_loader),
                    "total_loss",
                    total_loss.item()
                )
                cfg.log_string(msg)
                # iter += 1
                #print(msg)
            loss_dict={}
            for key in est_data:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    loss_dict[key] = est_data[key].item()
            for loss in loss_dict:
                if "total" not in loss:
                    tb_logger.add_scalar("train/" + loss, loss_dict[loss], iter)
            tb_logger.add_scalar("train/total_loss", total_loss.item(), iter)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar("train/lr", current_lr, iter)
            iter += 1
        if e%config['other']['test_interval']==0:
            ap_calculator = APCalculator(ap_iou_thresh=ap_iou_thresh,
                                         class2type_map=DC.class2type)
            model.eval()
            eval_loss = 0
            cfg.log_string("Switch Phase to Test")
            loss_info = {}
            for batch_id, data_batch in enumerate(test_loader):
                for key in data_batch:
                    if isinstance(data_batch[key], list) == False:
                        data_batch[key] = data_batch[key].cuda()
                with torch.no_grad():
                    est_data, loss = model(data_batch)
                total_loss = torch.mean(loss)
                msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    "epoch",
                    e,
                    batch_id + 1,
                    len(test_loader),
                    "test_loss",
                    total_loss.item()
                )
                for key in est_data:
                    if 'loss' in key or 'acc' in key or 'ratio' in key:
                        if key not in loss_info: loss_info[key]=0
                        loss_info[key] += est_data[key]

                batch_pred_map_cls = parse_predictions(est_data, CONFIG_DICT)
                batch_gt_map_cls = parse_groundtruths(est_data, CONFIG_DICT)
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

                eval_loss += total_loss.item()
                #cfg.log_string(msg)
                #print(msg)
            avg_eval_loss = eval_loss / (batch_id + 1)
            for key in loss_info:
                loss_info[key] = loss_info[key] / (batch_id + 1)
            print("eval_loss is", avg_eval_loss)
            tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)
            for key in loss_info:
                tb_logger.add_scalar("eval/" + key, loss_info[key], e)

            metrics_dict=ap_calculator.compute_metrics()
            for key in metrics_dict:
                cfg.log_string('eval %s: %f'%(key, metrics_dict[key]))

            checkpoint.register_modules(epoch=e, min_loss=avg_eval_loss)
            if avg_eval_loss < min_eval_loss:
                checkpoint.save('best')
                min_eval_loss = avg_eval_loss
            else:
                checkpoint.save("latest")
        scheduler.step()
        e += 1

def heatmap_trainer(cfg,model,optimizer,scheduler,train_loader,test_loader,checkpoint):
    start_t = time.time()
    config = cfg.config
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)
    start_epoch = 0
    if config["resume"] == True:
        checkpoint.load(config["weight"])
        start_epoch = scheduler.last_epoch
    scheduler.last_epoch = start_epoch
    model.train()
    iter = 0
    min_eval_loss = 10000
    for e in range(start_epoch, config['other']['nepoch']):
        cfg.log_string("Switch Phase to Train")
        model.train()
        for batch_id, data_batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            for key in data_batch:
                if isinstance(data_batch[key], list) == False:
                    data_batch[key] = data_batch[key].float().cuda()
            est_data, loss_dict = model(data_batch)
            total_loss = torch.mean(loss_dict["loss"])
            total_loss.backward()
            optimizer.step()
            if iter%config['other']['log_interval']==0:
                msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    "epoch",
                    e,
                    batch_id + 1,
                    len(train_loader),
                    "total_loss",
                    total_loss.item()
                )
                cfg.log_string(msg)
            # iter += 1
            for loss in loss_dict:
                if "total" not in loss:
                    tb_logger.add_scalar("train/" + loss, torch.mean(loss_dict[loss]).item(), iter)
            tb_logger.add_scalar("train/total_loss", total_loss.item(), iter)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar("train/lr", current_lr, iter)
            iter += 1
        model.eval()
        eval_loss = 0
        eval_loss_info = {
        }
        cfg.log_string("Switch Phase to Test")
        for batch_id, data_batch in enumerate(test_loader):
            for key in data_batch:
                if isinstance(data_batch[key], list) == False:
                    data_batch[key] = data_batch[key].float().cuda()
            with torch.no_grad():
                est_data, loss_dict = model(data_batch)
            total_loss = torch.mean(loss_dict["loss"])
            msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(test_loader),
                "test_loss",
                total_loss.item()
            )
            for key in loss_dict:
                if "total" not in key:
                    if key not in eval_loss_info:
                        eval_loss_info[key] = 0
                    eval_loss_info[key] += torch.mean(loss_dict[key]).item()

            total_loss = torch.mean(total_loss)
            eval_loss += total_loss.item()
            cfg.log_string(msg)
        avg_eval_loss = eval_loss / (batch_id + 1)
        for key in eval_loss_info:
            eval_loss_info[key] = eval_loss_info[key] / (batch_id + 1)
        print("eval_loss is", avg_eval_loss)
        tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)
        for key in eval_loss_info:
            tb_logger.add_scalar("eval/" + key, eval_loss_info[key], e)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_eval_loss)
        else:
            scheduler.step()

        checkpoint.register_modules(epoch=e, min_loss=avg_eval_loss)
        if avg_eval_loss < min_eval_loss:
            checkpoint.save('best')
            min_eval_loss = avg_eval_loss
        else:
            checkpoint.save("latest")
        e += 1