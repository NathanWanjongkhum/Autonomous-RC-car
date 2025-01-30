# Calibrate he camera
# Create an epipolar scheme
# Buiold a disparity map and depth map

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
class DepthMap:
    def __init__(self):
        root=os.getcwd()
        
        imgLeftPath = os.path.join(root, './images/left.png')
        imgRightPath = os.path.join(root, './images/right.png')
        self.imgLeft = cv.imread(imgLeftPath, cv.IMREAD_GRAYSCALE)
        self.imgRight = cv.imread(imgRightPath, cv.IMREAD_GRAYSCALE)
        
    def compute(self):
        # finds the disparity image
        # creates StereoBm object  
        stereo = cv.StereoBM_create(numDisparities = 16, 
                            blockSize = 15) 
  
        # computes disparity 
        disparity = stereo.compute(self.imgLeft, self.imgRight)
        
        # displays image as grayscale and plotted 
        plt.imshow(disparity, 'gray') 
        plt.show()


if __name__ == '__main__':
    map = DepthMap()
    map.compute()