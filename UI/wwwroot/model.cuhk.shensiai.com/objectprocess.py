import os
import csv
import json

def offToObj(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines)>2 and lines[0].strip()=='OFF':
            [vertex_num, surface_num, other_num] = lines[1].strip().split(' ')
            vset = lines[2:int(vertex_num)+2]
            fset = lines[int(vertex_num)+2:int(vertex_num)+3+int(surface_num)]
            with open(filename.replace('.off','.obj'), 'w') as out:
                for v in vset:
                    out.write('v '+v.strip()+' \n')
                out.write('\n')
                for f in fset:
                    params = f[1:].strip().split(' ')
                    out.write('f ')
                    for param in params:
                        index = int(param)+1
                        out.write(str(index)+' ')
                    out.write('\n')

def convertOff():
    path_off = 'F:/modelnet'
    for root, dirs, files in os.walk(path_off):
        for file in files:
            if file.endswith('off'):
                print(file)
                offToObj(os.path.join(root, file))

def uploadObject():
    with open('objectlist.json', 'w') as out:
        # upload ShapeNet
        root = 'static/models/ShapeNet'
        typelist = os.listdir(root)
        for type in typelist:
            objlist = os.listdir(os.path.join(root,type))
            for obj in objlist:
                path = os.path.join(root, type, obj, 'models', 'model_normalized.obj')
                if os.path.exists(path):
                    out.write('{"name":"%s","path":"/static/models/ShapeNet/%s/%s/models/model_normalized.obj","type":"%s"}\n'%(obj,type,obj,type))

        # upload ModelNet
        root = 'static/models/ModelNet'
        typelist = os.listdir(root)
        for type in typelist:
            objlist = os.listdir(os.path.join(root, type))
            for obj in objlist:
                path = os.path.join(root, type, obj, obj+'.obj')
                if os.path.exists(path):
                    out.write('{"name":"%s","path":"/static/models/ModelNet/%s/%s/%s.obj","type":"%s"}\n'%(obj,type,obj,obj,type))


def initObjectType():
    with open('F:/class.csv', 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        print(headers)
        type_list = []
        for i, rows in enumerate(f_csv):
            try:
                if rows[0]=='':
                    continue
                tables=[]
                for i, table in enumerate(rows):
                    if i>4:
                        if table == '1':
                            tables.append(headers[i])
                print({"source": rows[0], "type_id": rows[1], "type_name_en": rows[2],"type_name_cn": rows[2], "sort": int(rows[4]),
                                  "tables": tables})
                type_list.append({"source": rows[0], "type_id": rows[1], "type_name_en": rows[2],"type_name_cn": rows[2], "sort": int(rows[4]),
                                  "tables": tables})
            except:
                print('123')

if __name__ == '__main__':
    uploadObject()
