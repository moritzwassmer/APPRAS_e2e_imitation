import torch.nn as nn
import torchvision
import torch

import os

from torchvision.models import ResNet18_Weights

"""
These models are experimental changes to the architecture presented in
"architectures_v2" which in the end did not improve performance.
Two models are presented in this file. 
The first is very similar to "architectures_v2" but the nbr of output features
in the input layers is increased. 
The second has added dropout.  
"""

class Resnet_Baseline_V3(nn.Module):


    """ 
    Class implementing the ResNet18 architecture with added input
    layers and regression heads. This time the concatenated output 
    of (rgb, command, speed) is passed through an MLP in the forward pass.

    This model serves as as further experimentation on
    of the "architectures_v2".

    Attributes:
        net: vision backbone (ResNet)
        cmd_input: command input layer - one hot encoded vector
        spd_input: speed input layer
        mlp: MLP Module
        thr_head: Regression Heads for Throttle
        brk_head: Regression Heads for Brake
        str_head: Regression Heads for Steering
    """
    def __init__(self):
        super().__init__()

        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Identity()

        # Input Layer for cmd, spd
        self.cmd_input = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
        )

        self.spd_input = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_ftrs + 128 + 128, num_ftrs + 128 + 128),
            nn.ReLU(), 
            nn.Linear(num_ftrs + 128 + 128, num_ftrs + 128 + 128),
            nn.ReLU() 
        )

        # Regression Heads for Throttle, Brake and Steering
        self.thr_head = nn.Sequential(
            nn.Linear(num_ftrs + 128 + 128, 1),
            nn.Sigmoid()  # [0,1] Range Output

        )

        self.brk_head = nn.Sequential(
            nn.Linear(num_ftrs + 128 + 128, 1),
            nn.Sigmoid()  # [0,1] Range Output

        )

        self.str_head = nn.Sequential(
            nn.Linear(num_ftrs + 128 + 128, 1),
            nn.Tanh()  # [-1,1] Range Output

        )

    # Forward Pass of the Model
    def forward(self, rgb, cmd, spd):
        rgb = self.net(rgb)  # BRG
        cmd = self.cmd_input(cmd)
        spd = self.spd_input(spd)

        x = torch.cat((rgb, cmd, spd), 1)
        x = self.mlp(x)
        return self.brk_head(x), self.str_head(x), self.thr_head(x)
        

class Resnet_Baseline_V3_Dropout(nn.Module):

    """ 
    Class implementing the ResNet18 architecture with added input
    layers and regression heads. This time the concatenated output 
    of (rgb, command, speed) is passed through an MLP in the forward pass.

    This model serves as an experimentation with 
    "architectures_v2" containing dropout.

    Attributes:
        net: vision backbone (ResNet)
        cmd_input: command input layer - one hot encoded vector with 7 different navigational commands
        spd_input: speed input layer
        mlp: MLP Module
        thr_head: Regression Heads for Throttle
        brk_head: Regression Heads for Brake
        str_head: Regression Heads for Steering
    """
    def __init__(self, dropout_rate):
            super().__init__()

            # ResNet Architecture with pretrained weights, also bigger resnets available
            self.net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            num_ftrs = self.net.fc.in_features

            # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
            self.net.fc = nn.Sequential(
                nn.Identity(),
                nn.Dropout(p=dropout_rate, inplace=False)
            )

            # Input Layer for cmd, spd
            self.cmd_input = nn.Sequential(
                nn.Linear(7, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate, inplace=False)
            )

            self.spd_input = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate, inplace=False)
            )

            # MLP
            self.mlp = nn.Sequential(
                nn.Linear(num_ftrs + 128 + 128, num_ftrs + 128 + 128),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(num_ftrs + 128 + 128, num_ftrs + 128 + 128),
                nn.ReLU(), 
                nn.Dropout(p=dropout_rate, inplace=False)
            )

            # Regression Heads for Throttle, Brake and Steering
            self.thr_head = nn.Sequential(
                nn.Linear(num_ftrs + 128 + 128, 1),
                nn.Sigmoid()  # [0,1] Range Output

            )

            self.brk_head = nn.Sequential(
                nn.Linear(num_ftrs + 128 + 128, 1),
                nn.Sigmoid()  # [0,1] Range Output

            )

            self.str_head = nn.Sequential(
                nn.Linear(num_ftrs + 128 + 128, 1),
                nn.Tanh()  # [-1,1] Range Output

            )

    # Forward Pass of the Model
    def forward(self, rgb, cmd, spd):
        rgb = self.net(rgb)  # BRG
        cmd = self.cmd_input(cmd)
        spd = self.spd_input(spd)

        x = torch.cat((rgb, cmd, spd), 1)
        x = self.mlp(x)

        return self.brk_head(x), self.str_head(x), self.thr_head(x)