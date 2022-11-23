import numpy as np
import os
import glob
sigma=0.7

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

soft_heatmap=True
dataset_dir=r"C:\Users\LiuHaolin\Downloads\npz-TO-scannet\Volumes\ubuntu18\output\npz\TO-scannet"
save_dir=r"D:\TO-scannet-wHM"
error_log="./error_toscannet.txt"

if os.path.exists(save_dir)==False:
    os.makedirs(save_dir)
data_list=glob.glob(dataset_dir+"//*//*.npz")
data_list=data_list[:]
for filepath in data_list:
    filename=os.path.basename(filepath)
    #print("processing",filepath)
    split=filepath.split(os.sep)[-2]
    content=np.load(filepath)
    #print(list(content.keys()))
    xyz=content['xyz']
    color = content['color']

    if split!="test":
        bbox=content['bbox']
        instance_label=content['instance_label']
        semantic_label=content['semantic_label']
        heatmap=np.zeros((xyz.shape[0],1))
        if soft_heatmap:
            for i in range(bbox.shape[0]):
                dist = (xyz - bbox[i,0:3]) ** 2
                x_std = bbox[i,3]
                y_std = bbox[i,4]
                z_std = bbox[i,5]
                dist = dist / np.array([x_std ** 2, y_std ** 2, z_std ** 2])
                dist = np.sqrt(np.sum(dist, axis=1))
                gaussian_kernel = np.exp(-dist / 2 / sigma ** 2)
                cat_result=np.concatenate([heatmap,gaussian_kernel[:,np.newaxis]],axis=1)
                heatmap=np.max(cat_result,axis=1)[:,np.newaxis]
        nan_count=np.sum(np.isnan(heatmap).astype(np.float32))
        if nan_count>0:
            msg="nan value detected in %s"%(filename)
            print(msg)
            with open(error_log,'a') as f:
                f.write(msg)
        save_folder=os.path.join(save_dir,split)
        if os.path.exists(save_folder)==False:
            os.makedirs(save_folder)
        save_path=os.path.join(save_folder,filename)
        np.savez_compressed(save_path,heatmap=heatmap,xyz=xyz,bbox=bbox,color=color.astype(np.uint8),instance_label=instance_label.astype(np.uint8),semantic_label=semantic_label.astype(np.uint8))
    else:
        save_folder = os.path.join(save_dir, split)
        if os.path.exists(save_folder)==False:
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, filename)
        np.savez_compressed(save_path,xyz=xyz,color=color.astype(np.uint8))