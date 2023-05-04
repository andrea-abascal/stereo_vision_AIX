# Programa 1.1: Programa que verifica la conexión de la caámara y obtiene información acerca de ella

import cv2
import numpy as np

# Ask for camera index
i = int(input("Indica el índice de la cámara: "))

# Open stereo camera
cap = cv2.VideoCapture(i)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(width, height)


while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    
    
   # Display the resulting frame
   
    cv2.imshow('Imagen', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'): # wait for 's' key to save
        cv2.imwrite('Capture.png', frame)

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()


