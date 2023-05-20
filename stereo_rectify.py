import numpy as np
import cv2
import glob
import sys
import time


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()

cv_file.open('data/calibrationParameters.xml', cv2.FileStorage_READ)

newCameraMatrixL = cv_file.getNode('newCameraMatrixL').mat()
distL = cv_file.getNode('distL').mat()

newCameraMatrixR = cv_file.getNode('newCameraMatrixR').mat()
distR = cv_file.getNode('distR').mat()

rot = cv_file.getNode('rot').mat()
trans = cv_file.getNode('trans').mat()
essentialMatrix = cv_file.getNode('essentialMatrix').mat()
fundamentalMatrix = cv_file.getNode('fundamentalMatrix').mat()

grayL = cv_file.getNode('shape_L').mat()
grayR = cv_file.getNode('shape_R').mat()

cv_file.release()


rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)


print("Saving parameters!")

cv_file = cv2.FileStorage('data/stereoMap.xml', cv2.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])
cv_file.write('roi_L',roi_L)
cv_file.write('roi_R',roi_R)
cv_file.write('Q',Q)

cv_file.release()