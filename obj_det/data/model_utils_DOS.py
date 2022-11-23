# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os

class DOS_desk_config(object):
    def __init__(self):
        self.num_class = 52
        self.num_heading_bin = 1
        self.num_size_cluster = 52

        self.type2class = {"bag": 0, "bottle": 1, "bowl": 2, "camera": 3, "can": 4,
                      "cap": 5, "clock": 6, "keyboard": 7, "display": 8, "earphone": 9,
                      "jar": 10, "knife": 11, "lamp": 12, "laptop": 13, "microphone": 14,
                      "microwave": 15, "mug": 16, "printer": 17, "remote control": 18, "phone": 19,
                      "alarm": 20, "book": 21, "cake": 22, "calculator": 23, "candle": 24,
                      "charger": 25, "chessboard": 26, "coffee_machine": 27, "comb": 28, "cutting_board": 29,
                      "dishes": 30, "doll": 31, "eraser": 32, "eye_glasses": 33, "file_box": 34,
                      "fork": 35, "fruit": 36, "globe": 37, "hat": 38, "mirror": 39,
                      "notebook": 40, "pencil": 41, "plant": 42, "plate": 43, "radio": 44,
                      "ruler": 45, "saucepan": 46, "spoon": 47, "tea_pot": 48, "toaster": 49,
                      "vase": 50, "vegetables": 51}
        self.class2type = {self.type2class[t]: t for t in self.type2class}  # {0:'bag', ...}
        self.nyu40ids = np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                             59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                             76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93])
        self.nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))}
        self.mean_size_arr = np.load(os.path.join('./data/TO-crowd-means.npz'))['arr_0']
        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert (False)

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.

        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

class DOS_scene_config(object):
    def __init__(self):
        self.num_class = 70
        self.num_heading_bin = 1
        self.num_size_cluster = 70

        self.type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                      'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                      'refrigerator': 12, 'showercurtrain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16,
                      'garbagebin': 17,
                      "bag": 18, "bottle": 19, "bowl": 20, "camera": 21, "can": 22,
                      "cap": 23, "clock": 24, "keyboard": 25, "display": 26, "earphone": 27,
                      "jar": 28, "knife": 29, "lamp": 30, "laptop": 31, "microphone": 32,
                      "microwave": 33, "mug": 34, "printer": 35, "remote control": 36, "phone": 37,
                      "alarm": 38, "book": 39, "cake": 40, "calculator": 41, "candle": 42,
                      "charger": 43, "chessboard": 44, "coffee_machine": 45, "comb": 46, "cutting_board": 47,
                      "dishes": 48, "doll": 49, "eraser": 50, "eye_glasses": 51, "file_box": 52,
                      "fork": 53, "fruit": 54, "globe": 55, "hat": 56, "mirror": 57,
                      "notebook": 58, "pencil": 59, "plant": 60, "plate": 61, "radio": 62,
                      "ruler": 63, "saucepan": 64, "spoon": 65, "tea_pot": 66, "toaster": 67,
                      "vase": 68, "vegetables": 69}  # 把68 umbrella去掉了
        self.class2type = {self.type2class[t]: t for t in self.type2class}  # {0:'bag', ...}
        self.nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
                             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                             59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                             76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93])
        small_object_nyu40ids = np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                          59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                          76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93])
        self.nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))}
        self.mean_size_arr = np.load(os.path.join('./data/doscannet_means_scenelevel-relabel-5e6.npz'))['arr_0']
        #print(self.mean_size_arr.shape)
        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert (False)

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.

        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb


def rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))

    dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))

    for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:, 0] = crnr[0] * dx
        crnrs[:, 1] = crnr[1] * dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:, i] = crnrs[:, 0]
        new_y[:, i] = crnrs[:, 1]

    new_dx = 2.0 * np.max(new_x, 1)
    new_dy = 2.0 * np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)
