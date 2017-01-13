# Behavioral Cloning
This project is about cloning the human behavior in driving with a vehicle simulator using deep learning approach. [video](https://youtu.be/81VxK49Cers)

## Model Architecture Design
The model used for this project is the NVDIA's architecture with dropout at one of the fully connected layers. The number of neurons at the layer before dropout has been increased by 2 times to prevent underfitting. Exponential leaky units (ELU) is used for activation functions because it has smoother derivative a zero which will give better result for predicting the steering angles. For future work, I will try deeper network to improve the performance but for now, this is what my current hardware can handle. I have also tried using dropouts without increasing the number of neurons and it resulted in poorer performance. This shows that the performance will probably improve if more neurons and dropouts are used. The resulting model has also shown improvement in performance than the original NVDIA model (without dropouts).

## Architecture Characteristics
- Layer 1: Convolutional
    - 24 5x5 filters
    - Exponential leaky unit (ELU) activation
- Layer 2: Convolutional
    - 36 5x5 filters
    - Exponential leaky unit (ELU) activation
- Layer 3: Convolutional
    - 48 5x5 filters
    - Exponential leaky unit (ELU) activation
- Layer 4: Convolutional
    - 64 5x5 filters
    - Exponential leaky unit (ELU) activation
- Layer 5: Convolutional
    - 64 5x5 filters
    - Exponential leaky unit (ELU) activation
- Layer 6: Fully connected
    - 1164 neurons
    - Exponential leaky unit (ELU) activation
- Layer 7: Fully connected
    - 100 neurons
    - Exponential leaky unit (ELU) activation
- Layer 8: Fully connected
    - 50 neurons
    - Exponential leaky unit (ELU) activation
    - Dropout (0.4 dropout rate)
- Layer 9: Fully connected
    - 10 neurons
    - Exponential leaky unit (ELU) activation
<br>![image](https://github.com/yongkiat94/CarND-Behavioral-Cloning/blob/master/model.png)

# Data Preprocessing
Keras generator is used to augment the images so that augmented images will not be needed to be saved in disc and so that the data will only be read in by batches and not all at once.
For every images, the center, left, or right camera image will be selected at random. For left camera image, 0.25 will be added to the steering angle while for the right camera image, 0.25 will be subtracted from the steering angle.
The images are shifted to the left or right at random and added with 0.004 steering angle for every pixel shift to the right and subtracted with 0.004 steering angle for every pixel shift to the left.
It is then cropped by taking only 60 to 130 pixels of row so that the background distraction is minimized and the part of the car at the bottom is removed.
Lastly, the images are being flipped horizontally at random with negative of the steering angle if it is being flipped.
<br>Example training data:
<br>![image](https://github.com/yongkiat94/CarND-Behavioral-Cloning/blob/master/example.jpg)

# Model Training
The dataset is being splitted into 80% training data and 20% validation data.
The batch size for the model is 256 as this is the maximum size that I can use to train on my hardware. With batch size of 128, the loss oscillates even more and results in higher loss. The model is saved every 5 epochs and the final model is trained with 20 epochs as the loss seems to have converged.
Adam optimizer is used with learning rate of 0.001.