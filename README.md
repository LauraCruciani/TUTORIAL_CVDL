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

# TUTORIAL 3: Convolutional Neural Network (CNN) for Image Classification on CIFAR-10

This repository contains a tutorial on building a Convolutional Neural Network (CNN) using TensorFlow and Keras for classifying images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes.

---

## Overview

In this project, we:

- Load and preprocess the CIFAR-10 dataset.
- Normalize and standardize the data.
- Build a CNN model from scratch.
- Train the model.
- Evaluate the model on the test set.


## Dataset

We are using the **CIFAR-10** dataset, which contains:

- 50,000 training images
- 10,000 test images

The images are categorized into 10 different classes.
## Model Architecture

The CNN architecture consists of:
- **3 Convolutional Layers**: Each followed by Leaky ReLU activation, max-pooling, and dropout for regularization.
- **Fully Connected Layer**: Followed by dropout for regularization.
- **Output Layer**: A softmax activation function to output class probabilities for the 10 categories in the CIFAR-10 dataset.

The layers in the model are:

1. **Conv2D Layer 1**: 1024 filters, 3x3 kernel size, Leaky ReLU activation, followed by max-pooling and dropout (0.25).
2. **Conv2D Layer 2**: 512 filters, 3x3 kernel size, Leaky ReLU activation, followed by max-pooling and dropout (0.25).
3. **Conv2D Layer 3**: 256 filters, 3x3 kernel size, Leaky ReLU activation, followed by max-pooling and dropout (0.25).
4. **Fully Connected Layer**: 64 units, ReLU activation, followed by dropout (0.25).
5. **Output Layer**: 10 units with softmax activation.


# TUTORIAL 4 
# CNN Classification using Transfer Learning (Xception)

This script demonstrates the use of a pre-trained model, **Xception**, applied to the CIFAR-10 dataset through transfer learning. The goal is to solve a multiclass classification problem with 10 different classes.

---

## Preprocessing

1. **Normalization**: The CIFAR-10 images (32x32) are normalized to a range between 0 and 1.
2. **Standardization**: The dataset is standardized using the mean and standard deviation calculated from the training set.

## Transfer Learning with Xception
The model architecture is based on Xception pre-trained on ImageNet, with the following steps:

1. **UpSampling2D**: Since CIFAR-10 images are smaller than ImageNet images (224x224), we upsample them to 224x224.
2. **Pre-trained Xception**: The pre-trained Xception model is used as the feature extractor.
3. **Fully Connected Layer**: A dense layer with 128 units is added.
4. **Output Layer**: A softmax layer for 10-class classification.




