import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import cv2

data = pd.DataFrame()
path = 'data'
driving_log = pd.read_csv(path + '/driving_log.csv', skipinitialspace=True)
data['center'] = path + '/' + driving_log['center']
data['left'] = path + '/' + driving_log['left']
data['right'] = path + '/' + driving_log['right']
data['steering'] = driving_log['steering']

train, val = train_test_split(data[['center', 'left', 'right', 'steering']], random_state=0, test_size=0.3)

from scipy.misc import imresize
# Returns a cropped and resized image to be fed into neural network
def crop_image_and_resize(image):
    return imresize(image[60:130], (66, 200))

# Add 0.002 steering angle units per pixel shift to the right
# Subtracted 0.002 steering angle units per pixel shift to the left.
def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform() - trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 0
    trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, trans_M, (image.shape[1], image.shape[0]))
    
    return image_tr, steer_ang

def preprocess_image(data):
    # Randomly choose between center, left, or right image
    cam_clr = np.random.randint(3)
    if (cam_clr == 0):
        path_file = data['left']
        shift_ang = .25
    if (cam_clr == 1):
        path_file = data['center']
        shift_ang = 0.
    if (cam_clr == 2):
        path_file = data['right']
        shift_ang = -.25
    steering = data['steering'] + shift_ang
    
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, steering = trans_image(image, steering, 100)
    image = np.array(image)
    image = crop_image_and_resize(image)
    
    # Randomly flip images
    flip = np.random.randint(2)
    if flip == 1:
        image = np.fliplr(image)
        steering = -steering
    
    return image, steering

def generate_batch(data, batch_size=128):
    batch_images = np.zeros((batch_size, 66, 200, 3)).astype('uint8')
    batch_steering = np.zeros(batch_size)
    cur_index = 0
    while 1:
        for i in range(batch_size):
            if cur_index == len(data):
                cur_index = 0
                data = data.sample(frac=1).reset_index(drop=True)
            
            cur_data = data.iloc[cur_index]
            x, y = preprocess_image(cur_data)
            
            batch_images[i] = x
            batch_steering[i] = y
            cur_index += 1
        yield batch_images, batch_steering

def generate_val(data):
    images = np.zeros((len(data), 66, 200, 3)).astype('uint8')
    steerings = np.zeros(len(data))
    while 1:
        images = np.zeros((len(data), 66, 200, 3)).astype('uint8')
        steerings = np.zeros(len(data))
        for i in range(len(data)):
            images[i] = imresize(plt.imread(data.iloc[i]['center'])[60:130], (66, 200))
            steerings[i] = data.iloc[i]['steering']
        yield images, steerings

from keras.models import Sequential
from keras.layers.core import Lambda, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.layers import Dense

dropout = 0.2

def nvdia_model():
    model = Sequential()
    model.add(Lambda(lambda x : x/127.5 - 1., input_shape = (66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(dropout))
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, init='he_normal'))
    return model

model = nvdia_model()
# Using the default parameter for adam (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error',
              optimizer='adam')
model.fit_generator(generate_batch(train, 256), samples_per_epoch=5632, 
                    nb_epoch = 25, verbose=1, validation_data=generate_val(val), nb_val_samples=len(val))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")