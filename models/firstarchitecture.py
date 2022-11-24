#drafting the architecture inspired by nviida
#input: rgb
#output: steering angle

#IMPORTS

import os
import numpy as np
import csv
import cv2
import sklearn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch import nn
import torchvision.transforms.functional as fn

from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D




##some function for data extraction and downloading


#examples for collecting data from csv
'''
samples = []
with open('./data/data/driving_log.csv') as csvfile: #currently after extracting the file is present in this path
    reader = csv.reader(csvfile)
    next(reader, None) #this is necessary to skip the first record as it contains the headings
    for line in reader:
        samples.append(line)
'''
'''
samples = []

#load csv file
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)'''






train_samples, validation_samples = train_test_split(samples,test_size=0.15) 
#simply splitting the dataset to train and validation set usking sklearn. .15 indicates 15% of the dataset is validation set
# the 15% needs to be defined according to research, just random atm


#code for generator
#takes raw data and performs simple transformation (flip, angle augmentation)
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while True: 
        shuffle(samples) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size] 
            #handle that more data than rgb is collected

            images = []
            angles = []
            #check if we are collecting single camera image 
            for batch_sample in batch_samples:
                    name = 'format of file name tbd'
                    center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) #since CV2 reads an image in BGR we need to convert it to RGB since in drive.py it is RGB
                    center_angle = float(batch_sample[0]) #getting the steering angle measurement
                    images.append(center_image)
                    angles.append(center_angle)
                   
                   
                        
                        # Code for Augmentation of data.
                        # We take the image and just flip it and negate the measurement
                        
                    images.append(cv2.flip(center_image,1))
                    angles.append(center_angle*-1)
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) #here we do not hold the values of X_train and y_train instead we yield the values which means we hold until the generator is running
            

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#lambda pytorch version
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


model = nn.Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(LambdaLayer(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# trim image to only see section with road
#check how to actually do this in pytorch and crossref with needs
model.add(fn.crop(cropping=((70,25),(0,0))))           

#layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(nn.Conv2D(24,5,5,subsample=(2,2)))
model.add(nn.ELU)

#layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(nn.Conv2D(36,5,5,subsample=(2,2)))
model.add(nn.ELU)

#layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(nn.Conv2D(48,5,5,subsample=(2,2)))
model.add(nn.ELU)

#layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(nn.Conv2D(64,3,3))
model.add(nn.ELU)

#layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(nn.Conv2D(64,3,3))
model.add(nn.ELU)

#flatten image from 2D to side by side
model.flatten()

#layer 6- fully connected layer 1
model.add(Dense(100)) #It was at this point jakob realized writing in pytorch 
#is best to start over with a custom class altogether
model.add(nn.ELU)

#Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
model.add(Dropout(0.25))

#layer 7- fully connected layer 1
model.add(Dense(50))
model.add(nn.ELU)


#layer 8- fully connected layer 1
model.add(Dense(10))
model.add(nn.ELU)

#layer 9- fully connected layer 1
model.add(Dense(1)) #here the final layer will contain one value as this is a regression problem and not classification


# the output is the steering angle
# using mean squared error loss function is the right choice for this regression problem
# adam optimizer is used here
model.compile(loss='mse',optimizer='adam')


#fit generator is used here as the number of images are generated by the generator
# no of epochs : 5

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

#saving model
model.save('model.h5')

print('Done! Model Saved!')

# keras method to print the model summary
model.summary()