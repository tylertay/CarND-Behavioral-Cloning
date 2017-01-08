# Behavioral Cloning
This project is about cloning the human behavior in driving with a vehicle simulator using deep learning approach.

## Model Architecture
The model used for this project is the NVDIA's architecture with dropout at one of the fully connected layers. Exponential leaky units (ELU) is used for activation functions because it has smoother derivative a zero which will give better result for predicting the steering angles.
- Layer 1: Convolutional
* 24 5x5 filters
* Exponential leaky unit (ELU) activation
- Layer 2: Convolutional
* 36 5x5 filters
* Exponential leaky unit (ELU) activation
- Layer 3: Convolutional
* 48 5x5 filters
* Exponential leaky unit (ELU) activation
- Layer 4: Convolutional
* 64 5x5 filters
* Exponential leaky unit (ELU) activation
- Layer 5: Convolutional
* 64 5x5 filters
* Exponential leaky unit (ELU) activation
- Layer 6: Fully connected
* 1164 neurons
*  Exponential leaky unit (ELU) activation
* Dropout (0.2 dropout rate)
- Layer 7: Fully connected
* 100 neurons
*  Exponential leaky unit (ELU) activation
- Layer 8: Fully connected
* 50 neurons
*  Exponential leaky unit (ELU) activation
- Layer 9: Fully connected
* 10 neurons
*  Exponential leaky unit (ELU) activation

# Training
The dataset is being splitted into 70% training data and 30% validation data.
The batch size for the model is 256 and it is being trained for 25 epochs of 5625 training data.
Adam optimizer is used with learning rate of 0.001.
The model is trained using a keras generator to augment the images so that it does not have to save the augmented images.
For every images, the center, left, or right camera image will be selected at random. For left camera image, 0.25 will be added to the steering angle while for the right camera image, 0.25 will be subtracted from the steering angle.
The images are shifted to the left or right at random and added with 0.002 steering angle for every pixel shift to the right and subtracted with 0.002 steering angle for every pixel shift to the left.
It is then cropped by taking only 60 to 130 pixels of row so that the background distraction is minimized and the part of the car at the bottom is removed.
Lastly, the images are being flipped horizontally at random with negative of the steering angle if it is being flipped.
Example training data:
![image](https://github.com/yongkiat94/CarND-Behavioral-Cloning/blob/master/example.jpg)