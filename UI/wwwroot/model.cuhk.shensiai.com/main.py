# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import json
import numpy as np
from plyfile import PlyData, PlyElement
import requests

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def readJsonFile(path):
    file = open(path, "rb")
    return json.load(file)

path_ply = 'static/models/scenes'
path_np = '/data/items/scannet/scannet_train_detection_data'

def read_ply_xyzrgbal(filename):
    """ read XYZRGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 8], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
        vertices[:,6] = plydata['vertex'].data['alpha']
        vertices[:,7] = plydata['vertex'].data['label']
    return vertices, plydata['vertex'], plydata['face']

def scenePreprocess(scene_name):
    if os.path.exists(os.path.join(path_ply, scene_name, scene_name + '.ply')):
       return
    filename = os.path.join(path_ply, scene_name, scene_name + '_vh_clean_2.labels.ply')
    # 读取ply文件内容
    vertices, ply_vertex, ply_face = read_ply_xyzrgbal(filename)

    semantic_options = []
    if 3 in vertices[:, 7]:
        semantic_options.append(3)
    if 7 in vertices[:, 7]:
        semantic_options.append(7)
    if 12 in vertices[:, 7]:
        semantic_options.append(12)
    if 14 in vertices[:, 7]:
        semantic_options.append(14)

    print(semantic_options)

    # 读取npy文件内容
    npdata = np.load(os.path.join(path_np, scene_name + '_ins_label.npy'))
    # 读取json文件内容
    jsondata = readJsonFile(os.path.join(path_ply, scene_name, scene_name + '.aggregation.json'))
    segGroups = jsondata['segGroups']
    for seg in segGroups:
        seg['vertices'] = []
        seg['semantic'] = 0


    # 将每个table的顶点坐标存入json结构中
    for index, ver in enumerate(vertices):
        if ver[7] in semantic_options:
            segGroups[npdata[index] - 1]['vertices'].append(ver)
            if segGroups[npdata[index] - 1]['semantic'] == 0:
                segGroups[npdata[index] - 1]['semantic'] = ver[7]
            ply_vertex.data[index]['label'] = npdata[index]
        else :
            ply_vertex.data[index]['label'] = 0

    PlyData([ply_vertex, ply_face]).write(os.path.join(path_ply, scene_name, scene_name + '.ply'))

    # 计算所有table的中心点
    table_list = []
    for item in segGroups:
        if len(item['vertices']) > 0 and item['semantic']==3:
            points = np.array(item['vertices'])
            x_points = points[:, 0]
            y_points = points[:, 1]
            z_points = points[:, 2]
            table_list.append({'label': item['label'], 'id': item['id']+1, 'semantic': np.int( item['semantic']),
                               'x': np.float((x_points.min()+x_points.max())/2),
                               'y':  np.float((y_points.min()+y_points.max())/2),
                               'z': np.float(np.mean(z_points))})

    if len(table_list)>0:
        print(table_list)
        requests.post(
            "https://unicloud.shensiai.com/cuhk/model?action=updatescene&id=" + scene_name + '&param=' + json.dumps(
                table_list))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dirlist = os.listdir(path_ply)
    print('remove old ply')
    #for scene in dirlist:
    #    os.remove(os.path.join(path_ply, scene, scene + '.ply'))
    print('process start')
    for scene in dirlist:
        print(scene)
        scenePreprocess(scene)
