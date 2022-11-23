from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Dense, Flatten, Convolution2D
from sklearn.utils import shuffle
import numpy as np
import cv2

DATASET_PATH = '../dataset'

def image_generator(samples, batch_size=32):
  '''
  Python generator to generate/feed multiple batches of training-dataset.
  :param batch_size: Batch size of training-samples to generate.
  :return: Batch of features and labels, forming a part of the training-samples.
  '''
  num_samples = len(samples)
  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images = []
      angles = []
      for batch_sample in batch_samples:
        center_image_path = DATASET_PATH + '/IMG/' + batch_sample[0].split('/')[-1]
        center_image = cv2.cvtColor(cv2.imread(center_image_path), cv2.COLOR_BGR2RGB)
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)
      X_train = np.array(images)
      y_train = np.array(angles)
      yield shuffle(X_train, y_train)


def get_nvidia_end_to_end_model():
  '''
  Builds and returns the CNN model as discussed in nVidia's end-to-end pipeline paper.
  '''

  model = Sequential()

  # Cropping 70 pixel-rows from top and 25 pixel-rows from bottom.
  # Input: 3@160x320. Output: 3@65x320
  model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))  # Image dimension cropped down to 65x320.

  # Lambda layer to regularize the images so that their pixel-intensities are (mean) centered around zero with small standard-deviation.
  model.add(Lambda(lambda x : x / 127.5 - 1.))

  # Convolutional layer with ReLU activation. Input: 3@65x320. Output: 24@31x158.
  # Filter size : 5x5. Stride length : 2
  model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2,2)))

  # Convolutional layer with ReLU activation. Input: 24@31x158. Output: 36@14x77.
  # Filter size : 5x5. Stride length : 2
  model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2,2)))

  # Convolutional layer with ReLU activation. Input: 36@14x77. Output: 48@5x37.
  # Filter size : 5x5. Stride length : 2
  model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2,2)))

  # Convolutional layer with ReLU activation. Input: 48@5x37. Output: 64@4x36.
  # Filter size : 3x3. Stride length : 1
  model.add(Convolution2D(64, 3, 3, activation='relu'))

  # Convolutional layer with ReLU activation. Input: 64@4x36. Output: 64@3x35.
  # Filter size : 3x3. Stride length : 1
  model.add(Convolution2D(64, 3, 3, activation='relu'))

  model.add(Flatten())

  # Fully connected layer. Output: 100
  model.add(Dense(100))
  # Fully connected layer. Output: 50
  model.add(Dense(50))
  # Fully connected layer. Output: 10
  model.add(Dense(10))
  # Fully connected layer. Output: 1
  model.add(Dense(1))

  return model