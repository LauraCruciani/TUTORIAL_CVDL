import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import data, color, img_as_float, img_as_ubyte, filters, feature, io

### FEATURE DETECTION: EDGE AND CORNERS
##  EDGE DETECTION
# Load an input image and convert it to grayscale  
original_img = io.imread('data/images/castle.png')
# check if image is rgb or already in greyscale, convert to greyscale(in case its RGB),
# in both cases convert image to float
if(len(original_img.shape)==3):
    img_grey = img_as_float(color.rgb2gray(original_img))
elif(len(original_img.shape)==2):
    img_grey = img_as_float(original_img)
# Apply gaussian filter
sigma = np.sqrt(2) 
sigma2=np.sqrt(10)
img_smooth = filters.gaussian(img_grey, sigma)
img_smooth2 = filters.gaussian(img_grey, sigma2)
### VISUALIZE
plt.figure(figsize=(16,16))
plt.subplot(131)
plt.imshow(original_img, cmap=cm.gist_gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(img_smooth, cmap=cm.gist_gray)
plt.title('Gaussian filtered image. sigma=2')
plt.subplot(133)
plt.imshow(img_smooth2, cmap=cm.gist_gray)
plt.title('Gaussian filtered image, sigma 10')
plt.show()

### 1.2 Image gradient
#To estimate the first derivative (finite differences) of an image on the horizontal 
# (I_x) direction you can simply perform a convolution of the image with the kernel 
# k=[-0.5, 0, 0.5]. 
# We can try also with [-1,1]
# Partial derivatives kernel
k = np.array([-0.5, 0, 0.5]) 

# Compute first derivative along x
Ix = np.zeros_like(img_grey)
for i, r in enumerate(img_smooth2):
        Ix[i, :] = signal.convolve(r, k, mode='same')

# Compute first derivative along y
Iy = np.zeros_like(img_grey)
for i, c in enumerate(img_smooth2.T):
        Iy[:, i] = signal.convolve(c, k, mode='same')

# Compute the magnitude of the gradient
G = np.sqrt(Ix**2 + Iy**2)


k1=np.array([-1,1])
# Compute first derivative along x
Ix1 = np.zeros_like(img_grey)
for i, r in enumerate(img_smooth):
        Ix1[i, :] = signal.convolve(r, k1, mode='same')

# Compute first derivative along y
Iy1 = np.zeros_like(img_grey)
for i, c in enumerate(img_smooth.T):
        Iy1[:, i] = signal.convolve(c, k1, mode='same')

# Compute the magnitude of the gradient
G1 = np.sqrt(Ix1**2 + Iy1**2)

plt.figure(figsize=(12,6))

plt.subplot(231)
plt.imshow(Ix, cmap=cm.gist_gray)
plt.title(r'$I_x$')
plt.subplot(232)
plt.imshow(Iy, cmap=cm.gist_gray)
plt.title(r'$I_y$')
plt.subplot(233)
plt.imshow(G, cmap=cm.gist_gray)
plt.title(r'$G = \sqrt{I_x^2+I_y^2}$')
plt.subplot(234)
plt.imshow(Ix1, cmap=cm.gist_gray)
plt.title(r'$I_x$')
plt.subplot(235)
plt.imshow(Iy1, cmap=cm.gist_gray)
plt.title(r'$I_y$')
plt.subplot(236)
plt.imshow(G1, cmap=cm.gist_gray)
plt.title(r'$G = \sqrt{I_x^2+I_y^2}$')
plt.tight_layout
plt.show() 
# Thresholding the gradient you should be able to obtain the an estimate of the edges of the input image.
threshold = 0.03
edge = np.where(G1 > threshold, 1, 0)

plt.figure(figsize=(8,6))
plt.subplot(121)
plt.imshow(original_img, cmap=cm.gist_gray)
plt.title('Original image')
plt.subplot(122)
plt.imshow(edge, cmap=cm.gist_gray)
plt.title('Edge map');
plt.tight_layout()
plt.show



###### function
def edge_detector(img, sigma, threshold):

    """Perform edge detection on an input RGB image.
    
    - img: (n, m) input RGB image
    - sigma: float, standard deviation of the Gaussian smoothing
    - threshold: float, threshold value for the gradient
    """
    
    # Apply Gaussian filter
    img_smooth = filters.gaussian(img, sigma)
    
    # Compute first derivatives with the following kernel
    k = np.array([-0.5, 0, 0.5])

    # Compute first derivative along x
    Ix = np.zeros(img_smooth.shape)
    for i, r in enumerate(img_smooth):
        
        Ix[i, :] = signal.convolve(r, k, mode='same')

    # Compute first derivative along y
    Iy = np.zeros(img_smooth.shape)
    for i, c in enumerate(img_smooth.T):
        Iy[:, i] = signal.convolve(c, k, mode='same')

    # Compute the mangnitude of the gradient
    G = np.sqrt(Ix**2 + Iy**2)
    
    # Generate edge map
    edge = np.where(G > threshold, 255, 0)
    
    return edge


edge = edge_detector(img_grey,np.sqrt(2), 0.09)
edge1 = edge_detector(img_grey, np.sqrt(4), 0.09)
edge2 = edge_detector(img_grey, np.sqrt(10), 0.09)
edge3 = edge_detector(img_grey, np.sqrt(5), 0.01)
edge4 = edge_detector(img_grey, np.sqrt(5), 0.05)
edge5 = edge_detector(img_grey, np.sqrt(5), 0.15)
plt.figure(figsize=(12,8))
plt.subplot(321)
plt.imshow(edge, cmap=cm.gist_gray)
plt.title('Sigma 2, threshold 0.09')
plt.subplot(232)
plt.imshow(edge1, cmap=cm.gist_gray)
plt.title('Sigma 4, threshold 0.09')
plt.subplot(233)
plt.imshow(edge2, cmap=cm.gist_gray)
plt.title('Sigma 10, threshold 0.09')
plt.subplot(234)
plt.imshow(edge3, cmap=cm.gist_gray)
plt.title('Sigma 5, threshold 0.01')
plt.subplot(235)
plt.imshow(edge4, cmap=cm.gist_gray)
plt.title('Sigma 5, threshold 0.05')
plt.subplot(236)
plt.imshow(edge5, cmap=cm.gist_gray)
plt.title('Sigma 5, threshold 0.15')
plt.show()
