import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import tensorflowjs as tfjs

import cv2
import os
import glob

from PIL import Image

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D,UpSampling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.models import load_model
from keras import regularizers
from keras import backend as K


img_dir_train = "/home/srija/tensorflow/-45Images/train_images" # Enter Directory of all images 

data_path_train = os.path.join(img_dir_train,'*g')

num_train = 8
image_size = 256    
num_test = 80

files = glob.glob(data_path_train)
data = []
for f1 in files:
    image = cv2.imread(f1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (image_size,image_size))
    #img = np.array(img)
    #img = np.reshape(img, (28*28, 1))
    data.append(img)

train_data = data
train_data = np.array(train_data)   
print(train_data.shape)
train_data = np.reshape(train_data, (num_train, image_size, image_size))
print(train_data.shape)

td_random = []
for i in range(num_train):
    td = [[0]*image_size]*image_size
    idx = np.asscalar(np.random.randint(num_train, size=1))
    idx1 = np.asscalar(np.random.randint(num_train, size=1))
    idx2 = np.asscalar(np.random.randint(num_train, size=1))
    p = [[0]*image_size]
    p = (train_data[idx,:,:] + train_data[idx1,:,:] + train_data[idx2,:,:])/3
    # for j in range(image_size):
    #     for k in range(image_size):
            # print(j)
            # print(k)
            # print(idx)
            # print(train_data[idx][j][k])
            #td[j][k] = train_data[idx][j][k]
    td_random.append(p)
td_random = np.array(td_random)
print(td_random.shape)
td_random = np.reshape(td_random, (num_train,image_size,image_size))
print(td_random.shape)

print("Size of:")
print("- Training-set:\t\t{}".format(len(train_data)))
print("- Test-set:\t{}".format(len(td_random)))


input_img = Input(shape=(image_size, image_size, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(12, (5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(20, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(36, (5, 5), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(36, (5, 5), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(20, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(16, (5, 5), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
print(autoencoder.summary())
from tensorflow.python.keras.optimizers import Adam
optimizer = Adam(lr = 1e-3)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

train_data = train_data.astype('float32') / 255.
td_random = td_random.astype('float32') / 255.
train_data = np.reshape(train_data, (len(train_data), image_size, image_size, 1))  # adapt this if using `channels_first` image data format
td_random = np.reshape(td_random, (len(td_random), image_size, image_size, 1))  # adapt this if using `channels_first` image data format

from keras.callbacks import TensorBoard

autoencoder.fit(train_data, td_random,
                epochs=20,
                batch_size=3,
                shuffle=True,
                validation_data=(train_data, td_random))
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = train_data
decoded_imgs = autoencoder.predict(encoded_imgs)

n = num_train  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # save original
    img = train_data[i]
    img = cv2.resize(img, (image_size,image_size))
    img = np.array(img)
    img = np.reshape(img, (image_size*image_size, 1))
    #test_data[i] = np.reshape(test_data[i], (784,1))
    np.savetxt("/home/srija/tensorflow/-45Images/auto_output/ori/ori_"+str(i)+".txt", img)

    img = td_random[i]
    img = cv2.resize(img, (image_size,image_size))
    img = np.array(img)
    img = np.reshape(img, (image_size*image_size, 1))
    #test_data[i] = np.reshape(test_data[i], (784,1))
    np.savetxt("/home/srija/tensorflow/-45Images/auto_output/random/random_"+str(i)+".txt", img)


    # save reconstruction
    img = decoded_imgs[i]
    img = cv2.resize(img, (image_size,image_size))
    img = np.array(img)
    img = np.reshape(img, (image_size*image_size, 1))
    #decoded_imgs[i] = np.reshape(decoded_imgs[i], (784,1))
    np.savetxt("/home/srija/tensorflow/-45Images/auto_output/res/res_"+str(i)+".txt", img)
