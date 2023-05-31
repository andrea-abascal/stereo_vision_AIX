import sys
import numpy as np
import cv2


def getDepthMap(depthMap):
    depthMap[np.isnan(depthMap)] = 0 
    depthMap[np.isinf(depthMap)] = 0 
    depthMap = cv2.normalize(depthMap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


    return depthMap