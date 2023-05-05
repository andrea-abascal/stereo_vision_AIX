# Programa 1: Programa prueba con OpenCV con 2 camaras conectadas por USB. Abrir el Stream de las camaras y mostrar en 2 ventanas diferentes.

import cv2
import numpy as np
# Open stereo camera
capL = cv2.VideoCapture(2)
capL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
capR =cv2.VideoCapture(0)
capR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cv2.waitKey(2000)
#cap.set(3,640)
#cap.set(4,480)

while capR.isOpened() and capL.isOpened():
    # Capture frame-by-frame
    retR, frameR = capR.read()
    retL, frameL = capL.read()
    
    
      # Display the resulting frame
    cv2.namedWindow('Original Img', cv2.WINDOW_NORMAL)
    img = np.concatenate((frameL, frameR), axis = 1)
    cv2.imshow('Original Img', img)
    cv2.imshow('Left', frameL)
    cv2.imshow('Right', frameR)
    
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'): # wait for 's' key to save
        cv2.imwrite('Capture.png', frameL)

# Release and destroy all windows before termination

capR.release()
capL.release()
cv2.destroyAllWindows()
