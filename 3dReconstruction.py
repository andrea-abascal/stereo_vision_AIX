import numpy as np
import cv2
import time
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('data/stereoMap.xml', cv2.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
roi_L = cv_file.getNode('roi_L').mat()
roi_R= cv_file.getNode('roi_R').mat()
Q = cv_file.getNode('Q').mat()
cv_file.release()

def getDisparityVis(src: np.ndarray, scale: float = 1.0) -> np.ndarray:
    dst = (src * scale/16.0).astype(np.uint8)
    return dst

def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d,
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


def get_pts(infile):
	data = np.loadtxt(infile, delimiter=',')
	return data[12:,0], data[12:,1], data[12:,2] #returns X,Y,Z points skipping the first 12 lines
	
def plot_ply(infile):
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x,y,z = get_pts(infile)
	ax.scatter(x, y, z, c='r', marker='o')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()	

# Open both cameras
capL =cv2.VideoCapture(4)
capR = cv2.VideoCapture(2)

#cv2.namedWindow('Filtered Disparity',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Filtered Disparity',640,480)

# Select Block Matcher algorithm and open its corresponding parameters

cv_file_disp = cv2.FileStorage()
cv_file_disp.open('data/disparity_map_params.xml', cv2.FileStorage_READ)

# Creating an object of StereoBM algorithm
leftMatcher = cv2.StereoSGBM_create()

# Updating the parameters based on the saved data
numDisparities = cv_file_disp.getNode('numDisparities').real()
blockSize = cv_file_disp.getNode('blockSize').real()
preFilterCap = cv_file_disp.getNode('preFilterCap').real()
uniquenessRatio = cv_file_disp.getNode('uniquenessRatio').real()
speckleRange = cv_file_disp.getNode('speckleRange').real()
speckleWindowSize = cv_file_disp.getNode('speckleWindowSize').real()
disp12MaxDiff = cv_file_disp.getNode('disp12MaxDiff').real()
minDisparity = cv_file_disp.getNode('minDisparity').real()

cv_file_disp.release()

# Setting the updated parameters before computing disparity map
leftMatcher.setNumDisparities(int(numDisparities))
leftMatcher.setBlockSize(int(blockSize))
leftMatcher.setPreFilterCap(int(preFilterCap))
leftMatcher.setUniquenessRatio(int(uniquenessRatio))
leftMatcher.setSpeckleRange(int(speckleRange))
leftMatcher.setSpeckleWindowSize(int(speckleWindowSize))
leftMatcher.setDisp12MaxDiff(int(disp12MaxDiff))
leftMatcher.setMinDisparity(int(minDisparity))

rightMatcher = cv2.ximgproc.createRightMatcher(leftMatcher)

# Create WLS Filter
wlsFilter = cv2.ximgproc.createDisparityWLSFilter(leftMatcher)
lmbda = 8000
sigma = 1.5

# Set filter parameters
wlsFilter.setLambda(lmbda)
wlsFilter.setSigmaColor(sigma)


# Create varaibles for fps calculation
prevTime = 0
newTime = 0

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

while capR.isOpened() and capL.isOpened():
    # Capture frame-by-frame
    retR, frameR = capR.read()
    retL, cframeL = capL.read()
    
    imgR_gray = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(cframeL,cv2.COLOR_BGR2GRAY)
    
    # Undistort and rectify images
    frameR = cv2.remap(imgR_gray, stereoMapR_x, stereoMapR_y,cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frameL = cv2.remap(imgL_gray, stereoMapL_x, stereoMapL_y,cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    # Crop the image usion ROI
    xL, yL, wL, hL = roi_L
    xL = int(xL)
    yL = int(yL)
    wL = int(wL)
    hL = int(hL)
    frameL = frameL[yL:yL+hL, xL:xL+wL]
   
    xR, yR, wR, hR = roi_R
    xR = int(xR)
    yR = int(yR)
    wR = int(wR)
    hR = int(hR)
    frameR = frameR[yR:yR+hR, xR:xR+wR]

    w = min(wL,wR)
    h = hL

    frameL = cv2.resize(frameL, (w,h),interpolation = cv2.INTER_AREA)

    frameR = cv2.resize(frameR, (w,h),interpolation = cv2.INTER_AREA)
   
  
    # Calculating disparity
    disparityL = leftMatcher.compute(frameL,frameR)
    disparityR = rightMatcher.compute(frameR,frameL)
    
    # Applying filter
    wlsDisparity = wlsFilter.filter(disparityL, frameL,disparity_map_right= disparityR)
    
    
    # Display fps
    newTime = time.time()
    fps = 1/(newTime - prevTime)
    prevTime = newTime
    fps_text = 'FPS: {:.2f}'.format(fps)
    print(fps_text)
  
    # Displaying the disparity map
    filteredDispVis= getDisparityVis(wlsDisparity, 1)
    #cv2.putText(filteredDispVis, fps_text, (7,70), font, 1, (100, 255, 0), 1)
    #cv2.imshow("Filtered Disparity", filteredDispVis)


    xyzMap = cv2.reprojectImageTo3D(wlsDisparity, Q)
    
    #reflect on x axis
    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    xyzMap = np.matmul(xyzMap,reflect_matrix)

    #extract colors from image
    colors = cv2.cvtColor(cframeL, cv2.COLOR_BGR2RGB)
    
    #filter by min disparity
    mask = wlsDisparity > wlsDisparity.min()
    out_points = xyzMap[mask]
    print(out_points.shape)
    out_colors = colors[mask]

    #filter by dimension
    idx = np.fabs(out_points[:,0]) < 4.5
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]

    write_ply('results/pointcloud.ply', out_points, out_colors)
    
    
    '''infile = get_pts('results/pointcloud.ply')

    plot_ply(infile)'''

    # Read the point cloud
    pcd = o3d.io.read_point_cloud('results/pointcloud.ply') 

    # Visualize the point cloud within open3d
    o3d.visualization.draw_geometries([pcd]) 
    
    '''vis.add_geometry(pcd)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()'''
               
    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

# Release and destroy all windows before termination
capR.release()
capL.release()
cv2.destroyAllWindows()
# Close the visualization window
vis.destroy_window()