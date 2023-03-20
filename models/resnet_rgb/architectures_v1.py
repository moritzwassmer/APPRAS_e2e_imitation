import torch
import torch.nn as nn
import torchvision


class Resnet_Baseline_V1(nn.Module):
    

    """ 
    Class implementing the ResNet18 architecture with added input
    layers and regression heads.

    This model serves as a first attempt to use the ResNet architecture for our experiment.

    Attributes:
        net: vision backbone (ResNet)
        cmd_input: command input layer - one hot encoded vector with 7 different navigational commands
        spd_input: speed input layer
        thr_head: Regression Heads for Throttle
        brk_head: Regression Heads for Brake
        str_head: Regression Heads for Steering
    """


    def __init__(self):
        super().__init__()
        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet18(pretrained=True) # weights=True
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Identity()
        
        # Input Layer for cmd, spd
        self.cmd_input = nn.Sequential(
            nn.Linear(7, 7),
            nn.LeakyReLU() 
        )
        
        self.spd_input = nn.Sequential(
            nn.Linear(1, 1),
            nn.LeakyReLU() 
        )
        
        # Regression Heads for Throttle, Brake and Steering
        self.thr_head = nn.Sequential(
            nn.Linear(num_ftrs+8, 1),
            nn.Sigmoid() # [0,1] Range Output
        )
        
        self.brk_head = nn.Sequential(
            nn.Linear(num_ftrs+8, 1),
            nn.Sigmoid() # [0,1] Range Output
        )
        
        self.str_head = nn.Sequential(
            nn.Linear(num_ftrs+8, 1),
            nn.Tanh() # [-1,1] Range Output
        )

    # Forward Pass of the Model
    def forward(self, rgb, cmd, spd): # Sorting!
        rgb = self.net(rgb) # BRG
        cmd = self.cmd_input(cmd)
        spd = self.spd_input(spd)
        
        x = torch.cat((rgb, cmd, spd),1)
        
        return self.brk_head(x), self.str_head(x), self.thr_head(x) # Sorting!
