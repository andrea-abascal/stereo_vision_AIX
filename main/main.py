import sys
import cv2
import open3d as o3d

# Import user functions
from rectify_undistort import rectify_undistort as imgPreprocess
from fps_calculation import fps_calculation as fps
import disparity 
from depth import getDepthMap
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

#cv2.namedWindow('Filtered Disparity',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Filtered Disparity',640,480)


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

vis = o3d.visualization.Visualizer() 

num = 8
while capR.isOpened() and capL.isOpened():
    '''
    # Capture frame-by-frame
    ret, frame_L = capL.read()
    ret, frameR = capR.read()

    imgR_gray = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(frame_L,cv2.COLOR_BGR2GRAY)
    

    # Rectify and undistort frames
    frameL , frameR , w, h = imgPreprocess(imgL_gray,imgR_gray, stereoMapL_x, stereoMapL_y,stereoMapR_x, stereoMapR_y, roi_L, roi_R)
    frame_L = cv2.resize(frame_L, (w,h),interpolation = cv2.INTER_AREA)

    ''' 
    # Capture frame-by-frame
    retR, frame_R = capR.read()
    retL, frame_L = capL.read()
    
    imgR_gray = cv2.cvtColor(frame_R,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(frame_L,cv2.COLOR_BGR2GRAY)
    
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
    frame_L = cv2.resize(frame_L, (w,h),interpolation = cv2.INTER_AREA)
    frameR = cv2.resize(frameR, (w,h),interpolation = cv2.INTER_AREA)
    
    cv2.imshow("Step 1: Scene", frame_L)


    # Calculating disparity
    disparityL = leftMatcher.compute(frameL,frameR)
    disparityR = rightMatcher.compute(frameR,frameL)
    
    # Applying filter
    wlsDisparity = wlsFilter.filter(disparityL, frameL,disparity_map_right= disparityR)

    # Displaying the disparity map
    filteredDispVis= disparity.getDisparityVis(wlsDisparity, 1)
    cv2.imshow("Step 2: Filtered Disparity", filteredDispVis)
    
    xyzMap = cv2.reprojectImageTo3D(filteredDispVis, Q)

    depthMap = getDepthMap(xyzMap)
    depthVis = cv2.applyColorMap(depthMap, cv2.COLORMAP_SUMMER)
    cv2.imshow("Step 3: Depth Map", depthVis) 
    
    # Compute fps
    '''fps = fps()
    print(fps)'''
    
    key = cv2.waitKey(1)
    # Hit "q" to close the window
    if  key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'): # wait for 's' key to save
        if num == 8:
            # Create a visualization window
                   
            vis.create_window('Step 4: Reconstructed Scene',569,466)
            view_control = vis.get_view_control()
        else:
            pass
        vis.clear_geometries()
        # 3D map and colors
        points, colors = pointcloud.points(xyzMap,frame_L,filteredDispVis)
        
        # Generate the point cloud
        pointcloud.write_ply('results/s'+str(num)+'pointcloud.ply', points, colors)
        
        # Read the point cloud
        pcd = o3d.io.read_point_cloud('results/s'+str(num)+'pointcloud.ply') 
        vis.add_geometry(pcd)        
        
        view_control.set_zoom(0.0002)
        view_control.set_up([ -0.0023108895800350508, -0.98944745178302018, 0.14487373795632214 ])
        view_control.set_lookat([53.331142545750687, 26.46709928743525, 68.584828769678921 ])
        view_control.set_front([-0.031372204363311264, -0.14487454624381371, -0.98895255227135936])
        vis.run()

        cv2.imwrite('reconstruction/s'+str(num)+'depth.png',depthVis)
        
        cv2.imwrite('reconstruction/s'+str(num)+'disparity.png',filteredDispVis)
        
        cv2.imwrite('reconstruction/s'+str(num)+'scene.png',frame_L)
    
        vis.capture_screen_image('reconstruction/s'+str(num)+'pointcloud.png')
       
        print("Reconstruction data saved!: Scene",num)
        
        num+=1
        
    elif key & 0xFF == ord('k'):
        print('continue')
        
        pass
     
        

# Release and destroy all windows before termination
capR.release()
capL.release()
cv2.destroyAllWindows()
vis.destroy_window()
