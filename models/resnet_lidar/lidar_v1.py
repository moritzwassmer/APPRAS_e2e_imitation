import torch.nn as nn
import torchvision
import torch

import os

def load_weights(net,name):
    root = net.weights_folder
    path = os.path.join(root, name)
    net.load_state_dict(torch.load(path))
    return net

from torchvision.models import resnet18, ResNet18_Weights


## Adapted from models/resnet_baseline/architectures_v3.py, added lidar resnet in parallel
## "Middle Fusion"
class Resnet_Lidar_V1(nn.Module):

    def __init__(self):
        super().__init__()

        # RGB
        self.rgb_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs_rgb = self.rgb_net.fc.in_features
        self.rgb_net.fc = nn.Identity()

         # Lidar
        self.lidar_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs_lidar = self.lidar_net.fc.in_features
        self.lidar_net.fc = nn.Identity()


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
            nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, num_ftrs_rgb + num_ftrs_lidar + 128 + 128),
            nn.ReLU(), 
            nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, num_ftrs_rgb + num_ftrs_lidar + 128 + 128),
            nn.ReLU() 
        )

        # Regression Heads for Throttle, Brake and Steering
        self.thr_head = nn.Sequential(
            nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, 1),
            nn.Sigmoid()  # [0,1] Range Output

        )

        self.brk_head = nn.Sequential(
            nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, 1),
            nn.Sigmoid()  # [0,1] Range Output

        )

        self.str_head = nn.Sequential(
            nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, 1),
            nn.Tanh()  # [-1,1] Range Output

        )

    # Forward Pass of the Model
    def forward(self, rgb, lidar, cmd, spd):
        rgb = self.rgb_net(rgb)
        lidar = self.lidar_net(lidar) 
        cmd = self.cmd_input(cmd)
        spd = self.spd_input(spd)

        x = torch.cat((rgb, lidar, cmd, spd), 1)
        x = self.mlp(x)
        # x = self.net.fc(x)
        return self.brk_head(x), self.str_head(x), self.thr_head(x)  # TODO 3 Change new order
        

class Resnet_Lidar_V1_Dropout(nn.Module):

    def __init__(self, dropout_rate):
            super().__init__()

            # RGB
            self.rgb_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            num_ftrs_rgb = self.rgb_net.fc.in_features
            self.rgb_net.fc = nn.Sequential(
                nn.Identity(),
                nn.Dropout(p=dropout_rate, inplace=False)
            )

            # Lidar
            self.lidar_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            num_ftrs_lidar = self.lidar_net.fc.in_features
            self.lidar_net.fc = nn.Sequential(
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
                nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, num_ftrs_rgb + num_ftrs_lidar + 128 + 128),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, num_ftrs_rgb + num_ftrs_lidar + 128 + 128),
                nn.ReLU(), 
                nn.Dropout(p=dropout_rate, inplace=False)
            )

            # Regression Heads for Throttle, Brake and Steering
            self.thr_head = nn.Sequential(
                nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, 1),
                nn.Sigmoid()  # [0,1] Range Output

            )

            self.brk_head = nn.Sequential(
                nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, 1),
                nn.Sigmoid()  # [0,1] Range Output

            )

            self.str_head = nn.Sequential(
                nn.Linear(num_ftrs_rgb + num_ftrs_lidar + 128 + 128, 1),
                nn.Tanh()  # [-1,1] Range Output

            )

    # Forward Pass of the Model
    def forward(self, rgb, lidar, cmd, spd):
        rgb = self.rgb_net(rgb) 
        lidar = self.lidar_net(lidar) 
        cmd = self.cmd_input(cmd)
        spd = self.spd_input(spd)

        x = torch.cat((rgb, lidar, cmd, spd), 1)
        x = self.mlp(x)

        # x = self.net.fc(x)
        return self.brk_head(x), self.str_head(x), self.thr_head(x)  # TODO 3 Change new order
