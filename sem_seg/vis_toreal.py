import os
import numpy as np
import pickle

from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util import vis_util_real as vis_util


def get_color(i):
    ''' Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Args:
        index: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    '''
    b = (i) % 256  # least significant byte
    g = (i >> 8) % 256
    r = (i >> 16) % 256 # most significant byte
    return r, g, b


def main():
    classes = 53
    color_map = np.zeros((classes, 3))
    names = [line.rstrip('\n') for line in open('data/tovanilla_crowd_names.txt')]
    for i in range(classes):
        color_map[i, :] = get_color(i)
    exp_list = ['pointtransformer_desklevel_easy_final']
    data_root = '/remote-home/chenpei/data/output/real_scene/real_desklevel_easy_process_axisalign_fps3/npz'
    # data_list = sorted(os.listdir(data_root))
    # data_list = [item[:-4] for item in data_list if 'Area_' in item]
    data_list = [str(item) for item in range(0, 97)]
    intersection_meter, union_meter, target_meter = AverageMeter(), AverageMeter(), AverageMeter()

    print('<<<<<<<<<<<<<<<<< Start Evaluation <<<<<<<<<<<<<<<<<')
    test_area = [1]
    for i in range(len(test_area)):
        result_path = os.path.join('exp/toscan/pointtransformer_desklevel_easy_final/result')
        pred_save_folder = os.path.join(result_path, 'best_visual_real/pred')
        label_save_folder = os.path.join(result_path, 'best_visual_real/label')
        image_save_folder = os.path.join(result_path, 'best_visual_real/image')
        check_makedirs(pred_save_folder); check_makedirs(label_save_folder); check_makedirs(image_save_folder)
        with open(os.path.join(result_path, 'best_real/pred' + '.pickle'), 'rb') as handle:
            pred = pickle.load(handle)['pred']
        with open(os.path.join(result_path, 'best_real/label' + '.pickle'), 'rb') as handle:
            label = pickle.load(handle)['label']
        # data_split = [item for item in data_list if 'Area_{}'.format(test_area[i]) in item]
        data_split = data_list
        assert len(pred) == len(label) == len(data_split)
        for j in range(len(data_split)):
            print('processing [{}/{}]-[{}/{}]'.format(i+1, len(test_area), j+1, len(data_split)))
            data_name = data_split[j]
            data_path = os.path.join(data_root, 'id'+ data_name + '.npz')
            f = np.load(data_path)
            data = np.concatenate((f['xyz'], f['color'], np.expand_dims(f['semantic_label'], axis=-1), np.expand_dims(f['instance_label'], axis=-1)), axis=-1)  # npy, n*8
            coord, feat = data[:, :3], data[:, 3:6]
            pred_j, label_j = pred[j].astype(np.uint8), label[j].astype(np.uint8)

            ignore_idx = np.where(label_j == 255)
            pred_j[ignore_idx] = 255  # seg ignore_label to 255, which will be in original rgb during vis

            # pred_j_color, label_j_color = color_map[pred_j, :], color_map[label_j, :]
            vis_util.write_ply_color_rgb(coord, pred_j, feat, os.path.join(pred_save_folder, data_name +'.obj'))
            vis_util.write_ply_color_rgb(coord, label_j, feat, os.path.join(label_save_folder, data_name + '.obj'))
            vis_util.write_ply_rgb(coord, feat, os.path.join(image_save_folder, data_name + '.obj'))
            intersection, union, target = intersectionAndUnion(pred_j, label_j, classes, ignore_index=255)
            intersection_meter.update(intersection); union_meter.update(union); target_meter.update(target)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        print('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
