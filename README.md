# TUTORIAL 1
  ## Edge Detection and Feature Detection

This repository provides Python code for detecting edges in grayscale images. ù

### Overview

This project implements an edge detection method with the following steps:

1. **Convert Image to Grayscale**: Converts the input image to grayscale if it's in RGB format.
2. **Gaussian Smoothing**: Applies Gaussian filters with different sigma values to reduce noise.
3. **Gradient Computation**: Uses convolution to compute image gradients in both x and y directions.
4. **Edge Detection**: Combines gradients to detect edges and applies thresholding to highlight them.

# TUTORIAL 2
# Corner Detection: Shi-Tomasi Algorithm

This repository demonstrates the Shi-Tomasi corner detection algorithm using Python libraries such as `scikit-image` and `numpy`. The tutorial explores how to detect corners in grayscale images and compares two corner detection methods: naive thresholding and `skimage.feature.corner_peaks`. The effect of transformations (rotation, scaling) on corner detection is also visualized.

---

## Overview

This project implements corner detection on an input image with the following steps:

1. **Convert Image to Grayscale**: Convert the input image to grayscale if it's in RGB format.
2. **Shi-Tomasi Corner Detection**: Compute corner strength map using the Shi-Tomasi algorithm.
3. **Thresholding**: Extract corners by thresholding the corner strength map.
4. **Comparison**: Compare naive thresholding with `skimage.feature.corner_peaks`.
5. **Transformation**: Observe the effect of image transformations (rotation, scaling, polar warping) on corner detection.
The Shi-Tomasi corner detection algorithm generates a corner strength map, which can be thresholded or processed using corner_peaks to extract the detected corners. The code also demonstrates how image transformations such as rotation and scaling affect corner detection. Here's an example of the effect of rotating the image by 35 degrees and rescaling.

