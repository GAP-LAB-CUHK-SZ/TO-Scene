import open3d as o3d

def get_bbox(center_pred,size_pred,color=[1,0,0]):
    c_x,c_y,c_z=center_pred[0],center_pred[1],center_pred[2]
    s_x,s_y,s_z=size_pred[0],size_pred[1],size_pred[2]
    verts=[[c_x-s_x/2,c_y-s_y/2,c_z-s_z/2],
           [c_x-s_x/2,c_y-s_y/2,c_z+s_z/2],
           [c_x-s_x/2,c_y+s_y/2,c_z-s_z/2],
           [c_x-s_x/2,c_y+s_y/2,c_z+s_z/2],
           [c_x+s_x/2,c_y-s_y/2,c_z-s_z/2],
           [c_x+s_x/2,c_y-s_y/2,c_z+s_z/2],
           [c_x+s_x/2,c_y+s_y/2,c_z-s_z/2],
           [c_x+s_x/2,c_y+s_y/2,c_z+s_z/2]]

    lines=[[0,1],[0,2],[0,4],[1,3],
           [1,5],[2,3],[2,6],[3,7],
           [4,5],[4,6],[5,7],[6,7]]
    colors=[color for i in range(len(lines))]
    line_set=o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(verts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors=o3d.utility.Vector3dVector(colors)
    return line_set