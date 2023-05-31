import sys
import numpy as np
import cv2

def points(xyzMap, frameL,filteredDispVis):    
    
    #extract colors from image
    colors = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
    
    #filter by min disparity
    mask = filteredDispVis > filteredDispVis.min()
    out_points = xyzMap[mask]
    out_colors = colors[mask]

    return out_points, out_colors

def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1,3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num = len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt = ' %f %f %f %d %d %d')

