import os
import csv
import json
import numpy as np

def offToObj(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines)>2 and lines[0].strip()=='OFF':
            [vertex_num, surface_num, other_num] = lines[1].strip().split(' ')
            vset = lines[2:int(vertex_num)+2]
            fset = lines[int(vertex_num)+2:int(vertex_num)+3+int(surface_num)]
            with open(filename.replace('.off','.obj'), 'w') as out:
                vlist = []
                for v in vset:
                    vs = v.strip().split(' ')
                    vlist.append(list(map(float, vs)))

                vlist = np.array(vlist)
                x_points = vlist[:, 0]
                y_points = vlist[:, 1]
                z_points = vlist[:, 2]
                center = [x_points.mean(), y_points.mean(), z_points.mean()]
                vlist = vlist - center
                scale = x_points.max() - x_points.min()
                y_offset = y_points.max() - y_points.min()
                z_offset = z_points.max() - z_points.min()
                if scale <y_offset:
                    scale = y_offset
                if scale < z_offset:
                    scale = z_offset

                vlist = vlist/scale

                for v in vlist:
                    out.write('v {0} {1} {2} \n'.format(v[0],v[1],v[2]))
                out.write('\n')
                for f in fset:
                    params = f[1:].strip().split(' ')
                    out.write('f ')
                    for param in params:
                        index = int(param)+1
                        out.write(str(index)+' ')
                    out.write('\n')

def convertOff():
    path_off = 'static/models/ModelNet'
    for root, dirs, files in os.walk(path_off):
        for file in files:
            if file.endswith('off'):
                print(file)
                offToObj(os.path.join(root, file))


if __name__ == '__main__':
    convertOff()
