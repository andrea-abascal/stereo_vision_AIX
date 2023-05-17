import numpy as np
import cv2
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS (3D COORD) AND IMAGE POINTS (2D COORD) #############################

chessboardSize = (10,7) # Inner corners
frameSize = (640,480)  # Frame resolution

# Default from documentation when calibration should be ended
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm  # know the exact real distance in the object points

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


imagesLeft = sorted(glob.glob('images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('images/stereoRight/*.png'))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv2.imread(imgLeft)
    imgR = cv2.imread(imgRight)
    # pre process image to make code more faster by working with only one channel instead of three and find corners more easily
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize,  None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objpoints.append(objp)
        
        #get a more accurate estimate of corners

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv2.imshow('img left', imgL)
        cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv2.imshow('img right', imgR)
        cv2.waitKey(1000)


cv2.destroyAllWindows()

############## CALIBRATION AND UNDISTORTION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
print("Camera matrix : \n")
print(cameraMatrixL)

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
print("Camera matrix : \n")
print(cameraMatrixR)

########## Re-projection Error  #############################################
mean_errorL = 0
mean_errorR = 0
for i in range(len(objpoints)):
    imgpoints2L, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], cameraMatrixL, distL)
    errorL = cv2.norm(imgpointsL[i], imgpoints2L, cv2.NORM_L2)/len(imgpoints2L)
    mean_errorL += errorL
    imgpoints2R, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], cameraMatrixR, distR)
    errorR = cv2.norm(imgpointsR[i], imgpoints2R, cv2.NORM_L2)/len(imgpoints2R)
    mean_errorR += errorR
print( "total error Left: {}".format(mean_errorL/len(objpoints)) )
print( "total error Right: {}".format(mean_errorR/len(objpoints)) )

########## Stereo Vision Calibration #############################################
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


stereo_calib_flags = cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria = criteria_stereo, flags = stereo_calib_flags)

######################


cv_file = cv2.FileStorage('data/calibrationParameters.xml', cv2.FILE_STORAGE_WRITE)


cv_file.write('shape_R',grayR)
cv_file.write('shape_L',grayL)


cv_file.write('retStereo',retStereo)
cv_file.write('newCameraMatrixL',newCameraMatrixL)
cv_file.write('distL',distL)
cv_file.write('newCameraMatrixR',newCameraMatrixR)
cv_file.write('distR',distR)


cv_file.write('rot',rot)
cv_file.write('trans',trans)
cv_file.write('essentialMatrix',essentialMatrix)
cv_file.write('fundamentalMatrix',fundamentalMatrix)

cv_file.release()