import sys
import cv2


def rectify_undistort(cap, stereoMap_x, stereoMap_y, roi_1, roi_2):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    
    # Undistort and rectify images
    frame = cv2.remap(img_gray, stereoMap_x, stereoMap_y,cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    # Crop the image using ROI
    x, y, w1, h = roi_1
    x = int(x)
    y = int(y)
    w1 = int(w1)
    h = int(h)
    frame = frame[y:y+h, x:x+w1]
   
    x2, y2, w2, h2 = roi_2
    w2 = int(w2)
    
    w = min(w1,w1)
    h = h

    frame = cv2.resize(frame, (w,h),interpolation = cv2.INTER_AREA)

    return frame


   