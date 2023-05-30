import sys
import cv2


def rectify_undistort(frameL, frameR, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, roi_L, roi_R):

    img_grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
    img_grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)

    
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

    return frameL, frameR


   