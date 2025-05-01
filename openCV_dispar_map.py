import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# this is just checking that the sizes are the same and resizing the
# second one to work. 

print("Image1:")
image = cv.imread('lef2t.jpg', cv.IMREAD_GRAYSCALE)
height, width, channels = image.shape  if len(image.shape) == 3 else (*image.shape, 1)
print(f"Height: {height} pixels")
print(f"Width: {width} pixels")
print(f"Channels: {channels}")

print("Image2:")
image2 = cv.imread('right.png', cv.IMREAD_GRAYSCALE)
height, width, channels = image2.shape  if len(image2.shape) == 3 else (*image2.shape, 1)
print(f"Height: {height} pixels")
print(f"Width: {width} pixels")
print(f"Channels: {channels}")

image2Resized = cv.resize(image2, (3976, 3000))


# imgL = cv.imread('lef2t.jpg', 0)
# imgR = cv.imread('right.png', 0)


# stereo creates the map.
stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
#disparirty calls the stereo and then computes it with the two
# images. 
disparity = stereo.compute(image,image2Resized)
#shows the disparity map.
plt.imshow(disparity,'gray')
plt.show()



