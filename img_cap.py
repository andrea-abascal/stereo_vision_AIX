# Programa que permitir guardar las capturas de ambas camaras en sus folders respectivos. Este es un paso anterior al de la calibración.

import cv2
import numpy as np
# Open stereo camera

capL =cv2.VideoCapture(4)
capR = cv2.VideoCapture(2)
cv2.waitKey(2000)

num = 0
while capR.isOpened() and capL.isOpened():
    # Capture frame-by-frame
    retR, frameR = capR.read()
    retL, frameL = capL.read()
    
    
      # Display the resulting frame
    img = np.concatenate((frameL, frameR), axis = 1)
    cv2.imshow('Original Img', img)
    cv2.imshow('Left', frameL)
    cv2.imshow('Right', frameR)
    
    
    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'): # wait for 's' key to save
        
        cv2.imwrite('images/original/image'+str(num)+'.png',img)
        
        cv2.imwrite('images/stereoLeft/imageL'+str(num)+'.png',frameL)
        
        cv2.imwrite('images/stereoRight/imageR'+str(num)+'.png',frameR)
        
        print("images saved!")
        
        num+=1

# Release and destroy all windows before termination

capR.release()
capL.release()
cv2.destroyAllWindows()
