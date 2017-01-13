import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import cv2
from scipy.misc import imresize

# Path of the data files
path = 'data'
# Read driving log and append path to directory
data = pd.read_csv(path + '/driving_log.csv', skipinitialspace=True)
data['center'] = path + '/' + data['center']
data['left'] = path + '/' + data['left']
data['right'] = path + '/' + data['right']

# Split the data into 80% training and 20% validation set
train, val = train_test_split(data[['center', 'left', 'right', 'steering']], random_state=0, test_size=0.2)

# Returns a cropped and resized image to be fed into neural network
# Image is cropped to reduce redundant features like the background and the front of the car
def crop_image_and_resize(image):
    return imresize(image[60:130], (66, 200))

# Add 0.004 steering angle units per pixel shift to the right
# Subtracted 0.004 steering angle units per pixel shift to the left
def trans_image(image, steering, trans_range):
    # If trans_range = 100, range will be -50 to 50
    trans_val = trans_range * np.random.uniform() - trans_range / 2
    steering += 0.004 * trans_val
    trans_M = np.float32([[1, 0, trans_val], [0, 1, 0]])
    image_res = cv2.warpAffine(image, trans_M, (image.shape[1], image.shape[0]))
    return image_res, steering

# Augment the data and return
def preprocess_image(data):
    # Randomly choose between center, left, or right image
    cam_clr = np.random.randint(3)
    if (cam_clr == 0):
        image_dir = data['center']
        shift_angle = 0
    # Add 0.25 to steering angle if left image chosen and
    # Subtract 0.25 from steering angle if right image chosen
    if (cam_clr == 1):
        image_dir = data['left']
        shift_angle = 0.25
    if (cam_clr == 2):
        image_dir = data['right']
        shift_angle = -0.25
    steering = data['steering'] + shift_angle
    
    image = plt.imread(image_dir)
    # Translate the image 50 pixels to the left or right
    image, steering = trans_image(image, steering, 100)
    # Crop image and resize to be fed into network
    image = crop_image_and_resize(image)
    
    # Randomly flip images
    flip = np.random.randint(2)
    if flip == 1:
        image = np.fliplr(image)
        steering = -steering
    
    return image, steering

# Generator to generate a number of data according to batch_size parameter
def generate_batch(data, batch_size):
    features = []
    target = []
    input_count = 0
    while True:
        # Loop through all the data
        for i in range(len(data)):
            # Augment every data and get back the resulting image and steering angle
            cur_data = data.iloc[i]
            x, y = preprocess_image(cur_data)
            # Add data to list until number of data is the same as batch size
            if input_count < batch_size-1:
                features.append(x)
                target.append(y)
                input_count += 1
            # Number of data is the same as batch size, generate the batch of data
            else:
                features.append(x)
                target.append(y)
                yield np.array(features), np.array(target)
                # Clear the batch
                features = []
                target = []
                input_count = 0
        # If there are left over data, generate the remaining data
        # This occurs when num_data%batch_size != 0
        if input_count > 0:
            yield np.array(features), np.array(target)
            features = []
            target = []
            input_count = 0

# Generator to generate all the validation data
def generate_val(data):
    while 1:
        # Initialize or reset the array
        images = np.zeros((len(data), 66, 200, 3)).astype('uint8')
        steerings = np.zeros(len(data))
        # Loop through all the data and set them to images and steerings array
        for i in range(len(data)):
            # Crop and resize the image
            images[i] = crop_image_and_resize(plt.imread(data.iloc[i]['center']))
            # Get the steering angle for image
            steerings[i] = data.iloc[i]['steering']
        yield images, steerings

from keras.models import Sequential
from keras.layers.core import Lambda, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.layers import Dense

# Rate of dropout
dropout = 0.4

def nvdia_model():
    model = Sequential()
    # Input Layer.
    # Input: 66x200x3. Output: normalized input
    model.add(Lambda(lambda x : x/127.5 - 1., input_shape = (66, 200, 3)))
    # Layer 1: Convolutional Layer.
    # Input: 66x200x3. Output: 66x200x24.
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    # Layer 2: Convolutional Layer.
    # Input: 66x200x24. Output: 66x200x36.
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    # Layer 3: Convolutional Layer.
    # Input: 66x200x36. Output: 66x200x48.
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    # Layer 4: Convolutional Layer.
    # Input: 66x200x48. Output: 66x200x64.
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))
    model.add(ELU())
    # Layer 5: Convolutional Layer.
    # Input: 66x200x64. Ouput: 66x200x64.
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Flatten())
    # Layer 6: Fully Connected Layer.
    # Input: 844800. Output: 1164.
    model.add(Dense(1164, init='he_normal'))
    model.add(ELU())
    # Layer 7: Fully Connected Layer.
    # Input: 1164. Output: 100.
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    # Layer 8: Fully Connected Layer.
    # Input: 100. Output: 100.
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    # Layer 9: Fully Connected Layer. (with dropout rate of 0.4)
    # Input: 100. Output: 10.
    model.add(Dropout(dropout))
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    # Output Layer.
    model.add(Dense(1, init='he_normal'))
    return model

model = nvdia_model()
# Use adam optimizer with default parameters (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error',
              optimizer='adam')
model.fit_generator(generate_batch(train, 256), samples_per_epoch = len(train), 
                    nb_epoch=20, verbose=1, validation_data=generate_val(val), nb_val_samples=len(val))

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save weights
model.save_weights("model.h5")