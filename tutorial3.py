# Convolutional Neural Network (CNN) for classification
#we are going to work with cifar10, a  dataset consisting of 60000 32x32 colour images in 10 classes, 
# with 6000 images per class. There are 50000 training images and 10000 test images.
# The main goal of this laboratory is to solve a multiclass classification problem 
# with 10 different classes.

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

num_classes = 10
new_im_size = 32
channels = 3
cifar10 = tf.keras.datasets.cifar10
(x_learn, y_learn),(x_test, y_test) = cifar10.load_data()

## PREPROCESSING
# NORMALIZING THE DATA
print("Normalizing training set..")
x_learn = np.asarray(x_learn, dtype=np.float32) / 255 # Normalizing training set
print("Normalizing test set..")
x_test = np.asarray(x_test, dtype=np.float32) / 255 # Normalizing test set
## split
x_train, x_val, y_train, y_val = train_test_split(x_learn, y_learn, test_size=0.25, random_state=12)
# standardization: Another common practice in data pre-processing is standardization.
#The idea about standardization is to compute your dataset mean and standard deviation 
# in order to subtract from every data point $x$ the dataset mean $\mu$ and 
# then divide by the standard deviation $\sigma$.
# The outcome of this operation is to obtain a distribution with mean equal to 0 and 
# a standard deviation equal to 1.
# By applying normalization to our data we are making the features more similar to each other 
# and this usually makes the learning process easier.

def standardize_dataset(X):
    image_means = []
    image_stds = []

    for image in X:
        image_means.append(np.mean(image)) # Computing the image mean
        image_stds.append(np.std(image)) # Computing the image standard deviation

    dataset_mean = np.mean(image_means) # Computing the dataset mean
    dataset_std = np.mean(image_stds) # Computing the dataset standard deviation
    return [dataset_mean, dataset_std] # For every image we subtract to it the dataset mean and we divide by the dataset standard deviation

dataset_mean, dataset_std = standardize_dataset(x_train)

print("Standardizing training set..")
x_train = (x_train-dataset_mean)/dataset_std # Standardizing the training set
print("Standardizing validation set..")
x_val = (x_val-dataset_mean)/dataset_std # Standardizing the test set
print("Standardizing test set..")
x_test = (x_test-dataset_mean)/dataset_std # Standardizing the test set

# one hot encode target values
y_train_enc = tf.keras.utils.to_categorical(y_train)
y_val_enc = tf.keras.utils.to_categorical(y_val)
y_test_enc = tf.keras.utils.to_categorical(y_test)

print("Size of the training set")
print("x_train", x_train.shape)
print("y_train", y_train.shape)

print("Size of the validation set")
print("x_val", x_val.shape)
print("y_val", y_val.shape)

print("Size of the test set")
print("x_test", x_test.shape)
print("y_test", y_test.shape)

# 2.2 Training a model from scratch
# Now that we have pre-processed our data, we are going to create a convolutional model in Keras.
# Usually a convolutional model is made by two subsequent part:
# * A convolutional part
# * A fully connected

# Usually the convolutional part is made by some layers composed by
# * convolutional layer: performs a spatial convolution over images
# * pooling layer: used to reduce the output spatial dimension from $n$ to 1 by averaging the $n$
#  different value or considering the maximum between them
# * dropout layer: applied to a layer, consists of randomly "dropping out"

# The convolutional part produces its output and the fully connected part ties together the 
# received information in order to solve the classification problem.
# Let us start with a shallow architecture with only 2 conv

# Creating the model from scratch
import tensorflow.keras
from tensorflow.keras import Sequential,Input,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import accuracy_score


scratch_model = Sequential()

# Build here your keras model.

scratch_model.add(Conv2D(1024, kernel_size=(3, 3), padding='same',input_shape=(new_im_size, new_im_size, channels)))
scratch_model.add(LeakyReLU(alpha=0.1))
scratch_model.add(MaxPooling2D((2, 2),padding='same'))
scratch_model.add(Dropout(0.25))

scratch_model.add(Conv2D(512, kernel_size=(3, 3), padding='same',input_shape=(new_im_size, new_im_size, channels)))
scratch_model.add(LeakyReLU(alpha=0.1))
scratch_model.add(MaxPooling2D((2, 2),padding='same'))
scratch_model.add(Dropout(0.25))

scratch_model.add(Conv2D(256, kernel_size=(3, 3), padding='same',input_shape=(new_im_size, new_im_size, channels)))
scratch_model.add(LeakyReLU(alpha=0.1))
scratch_model.add(MaxPooling2D((2, 2),padding='same'))
scratch_model.add(Dropout(0.25))


scratch_model.add(Flatten())
scratch_model.add(Dense(64, activation='relu'))
scratch_model.add(Dropout(0.25))
scratch_model.add(Dense(10, activation='softmax'))

# Compile the model with the Adam optimizer
scratch_model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])

# Visualize the model through the summary function
scratch_model.summary()

def plot_history(history):

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


    
# Network parameters
batch_size = 64 # Setting the batch size
epochs = 5
# On GPU this is fast process
scratch_model_history = scratch_model.fit(x_train, y_train_enc, batch_size=batch_size, shuffle=True, epochs=epochs, validation_data=(x_val, y_val_enc))


plot_history(scratch_model_history)

print("Training accuracy: ", accuracy_score(np.argmax(scratch_model.predict(x_train), axis=-1), y_train))
print("Validation accuracy: ", accuracy_score(np.argmax(scratch_model.predict(x_val), axis=-1), y_val))
print("Test accuracy: ", accuracy_score(np.argmax(scratch_model.predict(x_test), axis=-1), y_test))
