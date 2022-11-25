#drafting the architecture inspired by nviida
#input: rgb
#output: steering angle

#IMPORTS


import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.module):
    def __init__(self) -> None:
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            #layer 1- Convolution: input: rgb image, no of output filters- 24, kernel size= 5x5, stride= 2x2
            #note that we process one image rather than three as input 
            nn.Conv2d(1, 24, 5, stride=2),
            nn.ELU(),
            #layer 2- Convolution: input: 24 layers, no of output filters- 36, kernel size= 5x5, stride= 2x2
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            #layer 3- Convolution: input: 36 layers, no of output filters- 48, kernel size= 5x5, stride= 2x2
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            #layer 4- Convolution: input: 48 layers, no of output filters- 64, kernel size= 3x3, stride= 1x1
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            #layer 5- Convolution: input: 64 layers, no of output filters- 64, kernel size= 3x3, stride= 1x1
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5) # remove half (Bernoulii distr) units to avoid overfitting
            #THIS IS A HYPERPARAMETER NOT SPECIFIED BY NVIDIA, CAN CHANGE
        )
        self.linear_layers = nn.Sequential(
            #flattens the total in features to vec of 100 elements
            #gain clarity on the input size here?
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            #flattens the total in features to vec of 50 elements
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            #flattens the total in features to vec of 10 elements
            nn.Linear(in_features=50, out_features=10),
            #flattens the total in features to vec of 1 element
            #here the final layer will contain one value as this is a regression problem and not classification
            nn.Linear(in_features=10, out_features=1)
        )
    
    def forward(self, x):
        """Forward pass."""
        input = input.view(input.size(0), 3, 70, 320)
        #gain clarity on the arguments here?
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


model = CNNModel()


