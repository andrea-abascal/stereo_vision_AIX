import sys
import numpy as np
import cv2

def getDisparityVis(src: np.ndarray, scale: float = 1.0) -> np.ndarray:
    dst = (src * scale/16.0).astype(np.uint8)
    return dst