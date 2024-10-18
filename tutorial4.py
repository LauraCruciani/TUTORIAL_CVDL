import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from tensorflow.keras import applications
import tensorflow
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
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
num_classes = 10
new_im_size = 32
channels = 3
cifar10 = tf.keras.datasets.cifar10
batch_size=64
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



model = models.Sequential()

model.add(tensorflow.keras.layers.UpSampling2D(size=(7,7),input_shape=(32,32,3)))

Xception_model = applications.Xception(weights = "imagenet", include_top=False, input_shape = (224, 224, channels))

for layer in Xception_model.layers:
    layer.trainable = False

Inputs = layers.Input(shape=(32,32,3))
x = model(Inputs)
x = Xception_model(x)
x = layers.Flatten()(x)
# let's add a fully-connected layer
x = layers.Dense(128, activation='relu')(x)
# and a logistic layer for 10 classes
predictions = layers.Dense(10, activation='softmax')(x)

# this is the model we will train
pre_trained_model = tensorflow.keras.Model(Inputs, outputs=predictions)
pre_trained_model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])
pre_trained_model.summary()
epochs = 5 #it may take a while, the number of epochs should be low...
pretrained_model_history = pre_trained_model.fit(x_train, y_train_enc, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val_enc))
plot_history(pretrained_model_history)

print("Training accuracy: ", accuracy_score(np.argmax(pre_trained_model.predict(x_train), axis=-1), y_train))
print("Validation accuracy: ", accuracy_score(np.argmax(pre_trained_model.predict(x_val), axis=-1), y_val))
print("Test accuracy: ", accuracy_score(np.argmax(pre_trained_model.predict(x_test), axis=-1), y_test))
