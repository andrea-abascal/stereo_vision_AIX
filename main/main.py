import sys
import cv2
import open3d as o3d

# Import user functions
import rectify_undistort as imgPreprocess
import fps_calculation as fps
import disparity 
import pointcloud

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

# Get parameters from block matcher algorithm 
cv_file_disp = cv2.FileStorage()
cv_file_disp.open('data/disparity_map_paramsSGBM.xml', cv2.FileStorage_READ)
numDisparities = cv_file_disp.getNode('numDisparities').real()
blockSize = cv_file_disp.getNode('blockSize').real()
preFilterCap = cv_file_disp.getNode('preFilterCap').real()
uniquenessRatio = cv_file_disp.getNode('uniquenessRatio').real()
speckleRange = cv_file_disp.getNode('speckleRange').real()
speckleWindowSize = cv_file_disp.getNode('speckleWindowSize').real()
disp12MaxDiff = cv_file_disp.getNode('disp12MaxDiff').real()
minDisparity = cv_file_disp.getNode('minDisparity').real()
cv_file_disp.release()

# Open both cameras
capL =cv2.VideoCapture(4)
capR = cv2.VideoCapture(2)

cv2.namedWindow('Filtered Disparity',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Filtered Disparity',640,480)


# Creating an object of StereoBM algorithm
leftMatcher = cv2.StereoSGBM_create()

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

while capR.isOpened() and capL.isOpened():
   
    # Rectify and undistort frames
    frameL = imgPreprocess(capL, stereoMapL_x, stereoMapL_y, roi_L, roi_R)
    frameR = imgPreprocess(capR, stereoMapR_x, stereoMapR_y, roi_R, roi_L)
   
    # Calculating disparity
    disparityL = leftMatcher.compute(frameL,frameR)
    disparityR = rightMatcher.compute(frameR,frameL)
    
    # Applying filter
    wlsDisparity = wlsFilter.filter(disparityL, frameL,disparity_map_right= disparityR)

    # Displaying the disparity map
    filteredDispVis= disparity.getDisparityVis(wlsDisparity, 1)
    cv2.imshow("Filtered Disparity", filteredDispVis)

    # 3D map and colors
    xyzMap = cv2.reprojectImageTo3D(wlsDisparity, Q)
    points, colors = pointcloud.points(xyzMap,frameL,filteredDispVis)
    
    # Generate the point cloud
    pointcloud.write_ply('results/pointcloud.ply', points, colors)
    
    # Read the point cloud
    pcd = o3d.io.read_point_cloud('results/pointcloud.ply') 

    # Visualize the point cloud within open3d
    o3d.visualization.draw_geometries([pcd]) 
    
    # Compute fps
    fps = fps()
    print(fps)
                     
    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

# Release and destroy all windows before termination
capR.release()
capL.release()
cv2.destroyAllWindows()