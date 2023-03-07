import torch.nn as nn
import torchvision
import torch

import os

from torchvision.models import ResNet18_Weights

"""
Those are models which are just saved for safety but were just used for tests which were mostly not successfull, in a sense that there
was a significant improvement
"""

class Resnet_Baseline_V3(nn.Module):

    def __init__(self):
        super().__init__()

        # ResNet Architecture with pretrained weights, also bigger resnets available
        # self.net = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Identity()

        # Input Layer fuer cmd, spd
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
        # x = self.net.fc(x)
        return self.brk_head(x), self.str_head(x), self.thr_head(x)
        

class Resnet_Baseline_V3_Dropout(nn.Module):

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

            # Input Layer fuer cmd, spd
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

        # x = self.net.fc(x)
        return self.brk_head(x), self.str_head(x), self.thr_head(x)