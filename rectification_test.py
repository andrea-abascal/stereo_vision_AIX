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
    
capL =cv2.VideoCapture(4)
capR = cv2.VideoCapture(2)
cv2.waitKey(1000)

while capR.isOpened() and capL.isOpened():
    # Capture frame-by-frame
    retR, frameR = capR.read()
    retL, frameL = capL.read()
    

    # Undistort and rectify images
    frameL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frameR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT, 0)
   
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
   
    #Visualize scanlines:
    cv2.line(frameR, (0,int(h/8)*1), (w,int(h/8)*1), (0, 255, 0) , 1)
    cv2.line(frameR, (0,int(h/8)*2), (w,int(h/8)*2), (0, 255, 0) , 1)
    cv2.line(frameR, (0,int(h/8)*3), (w,int(h/8)*3), (0, 255, 0) , 1)
    cv2.line(frameR, (0,int(h/8)*4), (w,int(h/8)*4), (0, 255, 0) , 1)
    cv2.line(frameR, (0,int(h/8)*5), (w,int(h/8)*5), (0, 255, 0) , 1)
    cv2.line(frameR, (0,int(h/8)*6), (w,int(h/8)*6), (0, 255, 0) , 1)
    cv2.line(frameR, (0,int(h/8)*7), (w,int(h/8)*7), (0, 255, 0) , 1)
    cv2.line(frameR, (0,int(h/8)*8), (w,int(h/8)*8), (0, 255, 0) , 1)


    cv2.line(frameL, (0,int(h/8)*1), (w,int(h/8)*1), (0, 255, 0) , 1)
    cv2.line(frameL, (0,int(h/8)*2), (w,int(h/8)*2), (0, 255, 0) , 1)
    cv2.line(frameL, (0,int(h/8)*3), (w,int(h/8)*3), (0, 255, 0) , 1)
    cv2.line(frameL, (0,int(h/8)*4), (w,int(h/8)*4), (0, 255, 0) , 1)
    cv2.line(frameL, (0,int(h/8)*5), (w,int(h/8)*5), (0, 255, 0) , 1)
    cv2.line(frameL, (0,int(h/8)*6), (w,int(h/8)*6), (0, 255, 0) , 1)
    cv2.line(frameL, (0,int(h/8)*7), (w,int(h/8)*7), (0, 255, 0) , 1)
    cv2.line(frameL, (0,int(h/8)*8), (w,int(h/8)*8), (0, 255, 0) , 1)


    # Display the resulting frame
    img = np.concatenate((frameL, frameR), axis = 1)
    cv2.imshow('Original Img', img)
    cv2.imshow('Left', frameL)
    cv2.imshow('Right', frameR)

    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('q'):
        break
        
       

# Release and destroy all windows before termination

capR.release()
capL.release()
cv2.destroyAllWindows()
