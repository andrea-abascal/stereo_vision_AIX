import sys
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

def rectify_undistort(img_grayL, img_grayR, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, roi_L, roi_R):

    # Undistort and rectify images
    frameL = cv2.remap(img_grayL, stereoMapL_x, stereoMapL_y,cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frameR = cv2.remap(img_grayR, stereoMapR_x, stereoMapR_y,cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    # Crop the image using ROI
    xL, yL, wL, hL = roi_L
    xL = int(xL)
    yL = int(yL)
    wL = int(wL)
    hL = int(h)
    frameL = frameL[yL:yL+hL, xL:xL+wL]
   
    xR, yR, wR, hR = roi_R
    xR = int(xR)
    yR = int(yR)
    wR = int(wR)
    hR = int(hR)

    w = min(wL,wR)
    h = hL

    frameL = cv2.resize(frameL, (w,h),interpolation = cv2.INTER_AREA)
    frameR = cv2.resize(frameR, (w,h),interpolation = cv2.INTER_AREA)

    return frameL, frameR , w ,h


   