import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

import cv2
import os
from os import listdir
import glob
import csv
from PIL import Image
import pandas as pd

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.models import load_model

path_to_csv_all = '/home/srija/tensorflow/HistoLogy_MNIST/hmnist_28_28_RGB.csv'

df = pd.read_csv(path_to_csv_all)
df = np.array(df)

total_num = 5000
num_categories = 8
num_each_cat = 625
test_num_cat = 25


train_num_cat = num_each_cat-test_num_cat
num_train = total_num - test_num_cat*num_categories
num_test = test_num_cat*num_categories


train_data = df[:,:-1]
train_labels = df[:,2352]
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = []
for i in range(0,8):
    if i == 0:
        test_data = train_data[range(num_each_cat*i,num_each_cat*i + test_num_cat)]
    else:
        frames = train_data[range(num_each_cat*i,num_each_cat*i + test_num_cat)]
        test_data = np.concatenate((test_data, frames), axis=0)
#test_data = train_data[range(0,25) + range]
#test_labels = train_labels[0:25]

test_labels = []
for i in range(0,8):
    if i == 0:
        test_labels = train_labels[range(num_each_cat*i,num_each_cat*i + test_num_cat)]
    else:
        frames = train_labels[range(num_each_cat*i,num_each_cat*i + test_num_cat)]
        test_labels = np.concatenate((test_labels, frames), axis=0)

test_labels = test_labels - min(test_labels)
test_labels = np.reshape(test_labels, (num_test,1))
test_labels = keras.utils.to_categorical(test_labels, num_classes=8)

data = []
for i in range(0,8):
    if i == 0:
        data = train_data[range(num_each_cat*i +test_num_cat,num_each_cat*(i + 1))]
    else:
        frames = train_data[range(num_each_cat*i +test_num_cat,num_each_cat*(i + 1))]
        data = np.concatenate((data, frames), axis=0)

train_data = data

labels = []
for i in range(0,8):
    if i == 0:
        labels = train_labels[range(num_each_cat*i +test_num_cat,num_each_cat*(i + 1))]
    else:
        frames = train_labels[range(num_each_cat*i +test_num_cat,num_each_cat*(i + 1))]
        labels = np.concatenate((labels, frames), axis=0)

train_labels = labels

#train_data = train_data[range(25,5000)]
#train_labels = train_labels[range(25,5000)]

train_labels = train_labels - min(train_labels)
train_labels = np.reshape(train_labels, (num_train,1))
train_labels_ori = train_labels
train_labels = keras.utils.to_categorical(train_labels, num_classes=8)

print(train_data.shape)
print(test_data.shape)
print(train_labels.shape)
print(test_labels.shape)


data = []
for i in range(num_train):
    t = train_data[i]
    t = np.array(t)
    #t = np.reshape(t, (28, 28, 3))
    #t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    t = np.resize(t, (784,1))
    #img = np.array(img)
    t = np.reshape(t, (28*28, 1))
    data.append(t)

train_data = np.array(data)

data = []
for i in range(num_test):
    t = test_data[i]
    t = np.array(t)
    #t = np.reshape(t, (28, 28, 3))
    #t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    t = np.resize(t, (784,1))
    #img = np.array(img)
    t = np.reshape(t, (28*28, 1))
    data.append(t)
test_data = np.array(data)

train_data = np.reshape(train_data, (num_train, 784))
test_data = np.reshape(test_data, (num_test, 784))

print(train_data.shape)
print(test_data.shape)
print(train_labels.shape)
print(test_labels.shape)

#print("Size of:")
#print("- Training-set:\t\t{}".format(len(train_data)))
#print("- Test-set:\t{}".format(len(test_data)))
#print("- Train-labels:\t\t{}".format(len(train_labels)))
#print("- Test-labels:\t\t{}".format(len(test_labels)))



img_size = 28

# 1 for grayscale
num_channels = 1

# The images are stored in one-dimensional arrays of this length.
img_size_flat = img_size*img_size*num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size, num_channels)

img_shape_full = (img_size, img_size, num_channels)

# Number of classes, one class for each of 10 digits.
num_classes = 8

model = Sequential()

model.add(InputLayer(input_shape = (img_size_flat, )))

model.add(Reshape(img_shape_full))

model.add(Conv2D(kernel_size = 5, strides = 1, filters = 32, padding = "same",
            activation = "relu", name = "conv_layer_1"))

model.add(MaxPooling2D(pool_size = 2, strides = 2))

model.add(Conv2D(kernel_size = 5, strides = 1, filters = 64, padding = "same",
            activation = "relu", name = "conv_layer_2"))

model.add(MaxPooling2D(pool_size = 2, strides = 2))

model.add(Conv2D(kernel_size = 5, strides = 1, filters = 80, padding = "valid",
            activation = "relu", name = "conv_layer_3"))

model.add(MaxPooling2D(pool_size = 2, strides = 2))


model.add(Flatten())

model.add(Dense(128, activation = "relu"))

model.add(Dense(num_classes, activation = "softmax"))

print(model.summary())

from tensorflow.python.keras.optimizers import Adam

optimizer = Adam(lr = 1e-3)

model.compile(optimizer = optimizer,
                loss = 'categorical_crossentropy',
                metrics = ["accuracy"])

model.fit(x = train_data,
            y = train_labels,
            epochs = 50, batch_size = 256)


result = model.evaluate(x = test_data,
                        y = test_labels)



for name,value in zip(model.metrics_names, result):
    print(name, value)

print("{0}: {1:.2}".format(model.metrics_names[1], result[1]))


y_pred = model.predict(test_data)

#print(y_pred)

cls_pred = np.argmax(y_pred, axis = 1)

print(cls_pred)
print(np.unique(train_labels_ori))
for i in range(0,8):
    print(train_labels[train_num_cat*(i+1)-1])

#for i in range(0,8):
 #   for j in range(test_num_cat*i, test_num_cat*(i+1)):
  #      print(cls_pred[j]) 



#np.savetxt('test_Labels_3000_2000.txt', cls_pred, delimiter=',')  
#np.savetxt('testLabels.txt', test_labels, delimiter=',')  
#np.savetxt('testLabels_ori.txt', np.unique(train_labels), delimiter=',')