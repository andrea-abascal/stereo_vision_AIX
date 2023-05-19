import numpy as np
import cv2
import time

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('data/stereoMap.xml', cv2.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
roi_L = cv_file.getNode('roi_L').mat()
roi_R= cv_file.getNode('roi_R').mat()
cv_file.release()

# Open both cameras
capL =cv2.VideoCapture(4)
capR = cv2.VideoCapture(2)

 
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',640,480)


algorithm = int(input('Select BM (1) or SGBM (2): '))
cv_file_disp = cv2.FileStorage()
if algorithm ==  1:
    cv_file_disp.open('data/disparity_map_params.xml', cv2.FileStorage_READ)
    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoBM_create()
else:
    cv_file_disp.open('data/disparity_map_paramsSGBM.xml', cv2.FileStorage_READ)
    # Creating an object of StereoSGBM algorithm
    stereo = cv2.StereoSGBM_create()

# Updating the parameters based on the saved info
numDisparities = cv_file_disp.getNode('numDisparities').real()
blockSize = cv_file_disp.getNode('blockSize').real()
preFilterCap = cv_file_disp.getNode('preFilterCap').real()
uniquenessRatio = cv_file_disp.getNode('uniquenessRatio').real()
speckleRange = cv_file_disp.getNode('speckleRange').real()
speckleWindowSize = cv_file_disp.getNode('speckleWindowSize').real()
disp12MaxDiff = cv_file_disp.getNode('disp12MaxDiff').real()
minDisparity = cv_file_disp.getNode('minDisparity').real()

if algorithm == 1 :
    preFilterType = cv_file_disp.getNode('preFilterType').real()
    preFilterSize = cv_file_disp.getNode('preFilterSize').real()
    textureThreshold = cv_file_disp.getNode('textureThreshold').real()
else:
    pass

cv_file_disp.release()

# Setting the updated parameters before computing disparity map
stereo.setNumDisparities(int(numDisparities))
stereo.setBlockSize(int(blockSize))
stereo.setPreFilterCap(int(preFilterCap))
stereo.setUniquenessRatio(int(uniquenessRatio))
stereo.setSpeckleRange(int(speckleRange))
stereo.setSpeckleWindowSize(int(speckleWindowSize))
stereo.setDisp12MaxDiff(int(disp12MaxDiff))
stereo.setMinDisparity(int(minDisparity))

if algorithm == 1 :
    stereo.setPreFilterType(int(preFilterType))
    stereo.setPreFilterSize(int(preFilterSize))
    stereo.setTextureThreshold(int(textureThreshold))
else:
    pass

prevTime = 0
newTime = 0

while capR.isOpened() and capL.isOpened():
    # Capture frame-by-frame
    retR, frameR = capR.read()
    retL, frameL = capL.read()
    
    imgR_gray = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
    
     # Undistort and rectify images
    frameR = cv2.remap(imgR_gray, stereoMapR_x, stereoMapR_y,cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frameL = cv2.remap(imgL_gray, stereoMapL_x, stereoMapL_y,cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
   # crop the image
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
   
    
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(frameL,frameR,cv2.CV_32F)

    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    #disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    #disparity = (disparity/16.0 - minDisparity)/numDisparities
    norm_coeff = 255/ disparity.max()

    # Display fps
    font = cv2.FONT_HERSHEY_SIMPLEX
    newTime = time.time()
    fps = 1/(newTime - prevTime)
    prevTime = newTime
    fps_text = 'FPS: {:.2f}'.format(fps)
    print(fps_text)
  
    # Displaying the disparity map
    #cv2.putText(disparity, fps_text, (7,70), font, 1, (100, 255, 0), 1)
    cv2.imshow("disp",disparity* norm_coeff/255)
                     
 
    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

# Release and destroy all windows before termination
capR.release()
capL.release()
cv2.destroyAllWindows()
