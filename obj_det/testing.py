import torch
import os
import datetime
import time
import numpy as np
import pickle
import torch.nn as nn

def heatmap_tester(cfg,model,loader,checkpoint):
    start_t = time.time()
    config = cfg.config
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    if config['resume'] == True:
        print("loading from",config['weight'])
        checkpoint.load(config['weight'])
    cfg.write_config()
    model.eval()
    eval_loss_info={}
    eval_loss=0
    for batch_id, data_batch in enumerate(loader):
        for key in data_batch:
            if isinstance(data_batch[key], list) == False:
                data_batch[key] = data_batch[key].float().cuda()
        with torch.no_grad():
            est_data, loss_dict = model(data_batch)
        total_loss = torch.mean(loss_dict["loss"])
        msg = "{:0>8},[{}/{}],{}: {}".format(
            str(datetime.timedelta(seconds=round(time.time() - start_t))),
            batch_id + 1,
            len(loader),
            "test_loss",
            total_loss.item()
        )
        for key in loss_dict:
            if "total" not in key:
                if key not in eval_loss_info:
                    eval_loss_info[key] = 0
                eval_loss_info[key] += torch.mean(loss_dict[key]).item()

        if config['other']['dump_result']:
            for i in range(est_data["pred_heatmap"].shape[0]):
                id=data_batch["id"][i]
                xyz=data_batch["point_clouds"][i].detach().cpu().numpy()
                pred_heatmap=est_data["pred_heatmap"][i].detach().cpu().numpy()
                gt_heatmap=data_batch["gt_heatmap"][i].detach().cpu().numpy()
                save_path=os.path.join(log_dir,id+"_result.npz")
                np.savez_compressed(save_path,point_clouds=xyz,pred_heatmap=pred_heatmap,gt_heatmap=gt_heatmap)

        total_loss = torch.mean(total_loss)
        eval_loss += total_loss.item()
        cfg.log_string(msg)
    avg_eval_loss = eval_loss / (batch_id + 1)
    for key in eval_loss_info:
        eval_loss_info[key] = eval_loss_info[key] / (batch_id + 1)
    print("eval_loss is", avg_eval_loss)