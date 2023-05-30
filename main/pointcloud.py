import sys
import numpy as np
import cv2

def points(xyzMap, frameL,filteredDispVis):    
    #reflect on x axis
    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    xyzMap = np.matmul(xyzMap,reflect_matrix)

    #extract colors from image
    colors = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
    
    #filter by min disparity
    mask = filteredDispVis > filteredDispVis.min()
    out_points = xyzMap[mask]
    out_colors = colors[mask]

    '''#filter by dimension
    idx = np.fabs(out_points[:,0]) < 4.5
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]
    '''
    return out_points, out_colors

def write_ply(fn, verts, colors):
    ply_header = '''ply,
    format ascii 1.0,
    element vertex %(vert_num)d,
    property float x,
    property float y,
    property float z,
    property uchar red,
    property uchar green,
    property uchar blue,
    end_header
    '''
    out_colors = colors.copy()
    out_colors = out_colors.reshape(-1,3)
    verts = verts.reshape(-1,3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num = len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt = ' %f %f %f %d %d %d')

