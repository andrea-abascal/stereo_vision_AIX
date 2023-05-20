import numpy as np
import cv2

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

def nothing(x):
    pass
 
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',604,436)
 
cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',5,25,nothing)
 
# Creating an object of StereoBM algorithm
stereo = cv2.StereoSGBM_create()

def getDisparityVis(src: np.ndarray, scale: float = 1.0) -> np.ndarray:
    '''Replicated OpenCV C++ function

    Found here: https://github.com/opencv/opencv_contrib/blob/b91a781cbc1285d441aa682926d93d8c23678b0b/modules/ximgproc/src/disparity_filters.cpp#L559
    
    Arguments:
        src (np.ndarray): input numpy array
        scale (float): scale factor

    Returns:
        dst (np.ndarray): scaled input array
    '''
    dst = (src * scale/16.0).astype(np.uint8)
    return dst



while capR.isOpened() and capL.isOpened():
    # Capture frame-by-frame
    retR, frameR = capR.read()
    retL, frameL = capL.read()
    
    imgR_gray = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
    
    # Undistort and rectify images
    frameR = cv2.remap(imgR_gray, stereoMapR_x, stereoMapR_y,cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frameL = cv2.remap(imgL_gray, stereoMapL_x, stereoMapL_y,cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
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
   

    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')
     
    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(frameL,frameR,cv2.CV_32F)
    '''# NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    #disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    #disparity = (disparity/16.0 - minDisparity)/numDisparities
    norm_coeff = 255/ disparity.max()

 
    # Displaying the disparity map
    cv2.imshow("disp",disparity* norm_coeff/255)'''

    # Displaying the disparity map
    filteredDispVis= getDisparityVis(disparity, 1)
    #cv2.putText(filteredDispVis, fps_text, (7,70), font, 1, (100, 255, 0), 1)
    cv2.imshow('disp', filteredDispVis)
                                      

    # Display the resulting frame
    img = np.concatenate((frameL, frameR), axis = 1)
    cv2.imshow('Original Img', img)

    # Hit "s" to save parameters and  close the window
    if cv2.waitKey(1) & 0xFF == ord('s'):
      print('Saving depth estimation parameters ......')

      cv_file = cv2.FileStorage('data/disparity_map_paramsSGBM.xml', cv2.FILE_STORAGE_WRITE)
      cv_file.write("numDisparities",numDisparities)
      cv_file.write("blockSize",blockSize)
      cv_file.write("preFilterCap",preFilterCap)
      cv_file.write("uniquenessRatio",uniquenessRatio)
      cv_file.write("speckleRange",speckleRange)
      cv_file.write("speckleWindowSize",speckleWindowSize)
      cv_file.write("disp12MaxDiff",disp12MaxDiff)
      cv_file.write("minDisparity",minDisparity)
      cv_file.write("M",39.075)
      cv_file.release()
      break
    
    # Hit "q" to close the window
    elif cv2.waitKey(1) & 0xFF == ord('q'):
      break
        


# Release and destroy all windows before termination
capR.release()
capL.release()
cv2.destroyAllWindows()
