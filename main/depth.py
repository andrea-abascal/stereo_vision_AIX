import sys
import numpy as np


def getDepthMap(focal, baseline, disparity):
    depthMap = int(focal * baseline) / (disparity)
    depthMap = depthMap - depthMap.min()
    depthMap = depthMap/ depthMap.max() # normalize the data to 0 - 1
    depthMap = 255 * depthMap # Now scale by 255
    depthMap = depthMap.astype(np.uint8)

    return depthMap