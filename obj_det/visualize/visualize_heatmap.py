import open3d as o3d
import numpy as np
from utils import get_bbox

filepath=r"D:/fsdownload/id3075_result.npz"
content=np.load(filepath)
print(list(content.keys()))
xyz=content['point_clouds'][:,0:3]
heatmap=content['pred_heatmap'][0]
gt_heatmap=content['gt_heatmap'][:,0]
print(xyz.shape,heatmap.shape,gt_heatmap.shape)
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(xyz)
color=np.zeros(xyz.shape)
color[:,0]=gt_heatmap[:]
color[:,1]=gt_heatmap[:]
color[:,2]=gt_heatmap[:]
pcd.colors=o3d.utility.Vector3dVector(color)

vis=o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
#vis.add_geometry(bbox_line)
opt=vis.get_render_option()
opt.background_color=(0.2,0.2,0.2)
vis.run()