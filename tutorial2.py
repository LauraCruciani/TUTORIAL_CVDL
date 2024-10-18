import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import data, color, img_as_float, img_as_ubyte, filters, feature, io
####### CORNER DETECTION
# ## 2. Corner detection: SHI-TOMASI ALGORITHM

original_img = io.imread('data/images/castle.png')
# check if image is rgb or already in greyscale, convert to greyscale(in case its RGB),
# in both cases convert image to float
if(len(original_img.shape)==3):
    img_grey = img_as_float(color.rgb2gray(original_img))
elif(len(original_img.shape)==2):
    img_grey = img_as_float(original_img)

corners_map = feature.corner_shi_tomasi(img_grey, np.sqrt(2))


plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(original_img, cmap=cm.gist_gray)
plt.title('Original image')
plt.subplot(122)
plt.imshow(corners_map, cmap=cm.jet)
plt.colorbar(orientation='horizontal')
plt.title('Corners map')
plt.show()
### TO EXTRACT THE CORNER WE CAN THRESHOLD OR USE corner_peaks
threshold = 0.09
naive_corners = np.where(corners_map > threshold)

plt.figure(figsize=(12,6))
plt.imshow(original_img, cmap=cm.gist_gray)
plt.scatter(naive_corners[1], naive_corners[0], s=1, c='r');
plt.show()

corners = feature.corner_peaks(feature.corner_shi_tomasi(img_grey),threshold_abs=threshold)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(original_img, cmap=cm.gist_gray)
plt.scatter(corners[:,1], corners[:,0], s=1, c='r')
plt.title('skimage.feature.corner_peaks result')

plt.subplot(122)
plt.imshow(original_img, cmap=cm.gist_gray)
plt.scatter(naive_corners[1], naive_corners[0], s=1, c='r')
plt.title('naive thresholding result');
plt.show()
##############
### EFFECT OF TRANSFORMATION
from skimage.transform import warp_polar, rotate, rescale
threshold=0.3

modified=rotate(original_img, 35)
modified=rescale(modified, 0.75)
modified=warp_polar(modified)

if(len(original_img.shape)==3):
    img_grey = color.rgb2gray(original_img)
elif(len(original_img.shape)==2):
    img_grey = original_img

rotated=rotate(img_grey, 35)

corners_map = feature.corner_shi_tomasi(img_grey, np.sqrt(2))
corners_map_rotated=feature.corner_shi_tomasi(rotated, np.sqrt(2))

corners = feature.corner_peaks(feature.corner_shi_tomasi(img_grey),threshold_abs=threshold)
corners_rotated=feature.corner_peaks(feature.corner_shi_tomasi(rotated),threshold_abs=threshold)

naive_corners = np.where(corners_map > threshold)
naive_corners_rotated = np.where(corners_map_rotated > threshold)
plt.figure(figsize=(12,6))
plt.subplot(221)
plt.imshow(img_grey, cmap=cm.gist_gray)
plt.scatter(corners[:,1], corners[:,0], s=np.sqrt(2), c='r')
plt.title('skimage.feature.corner_peaks result')

plt.subplot(222)
plt.imshow(img_grey, cmap=cm.gist_gray)
plt.scatter(naive_corners[1], naive_corners[0], s=np.sqrt(2), c='r')
plt.title('naive thresholding result');
plt.subplot(223)
plt.imshow(rotated, cmap=cm.gist_gray)
plt.scatter(corners_rotated[:,1], corners_rotated[:,0], s=np.sqrt(2), c='r')
plt.title('skimage.feature.corner_peaks result')

plt.subplot(224)
plt.imshow(rotated, cmap=cm.gist_gray)
plt.scatter(naive_corners_rotated[1], naive_corners_rotated[0], s=np.sqrt(2), c='r')
plt.title('naive thresholding result');
plt.show()
