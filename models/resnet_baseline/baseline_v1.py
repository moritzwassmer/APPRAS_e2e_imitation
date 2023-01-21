import torch
import torch.nn as nn
import torchvision

"""
Input: rgb, speed, navigational command.
Output: break, steer, throttle.
Seq len: 1
Fusion: late fusion (concatenating).
Comment: no dense layers after concatenation. 
"""

class Baseline_V1(nn.Module):
    
    def __init__(self):
        super().__init__()
        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet18(pretrained=True) # weights=True
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Identity()
        
        # Input Layer fuer cmd, spd
        self.cmd_input = nn.Sequential(
            nn.Linear(7, 7),
            nn.LeakyReLU() # TODO
        )
        
        self.spd_input = nn.Sequential(
            nn.Linear(1, 1),
            nn.LeakyReLU() # TODO
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
        # print(rgb.shape, cmd.shape, spd.shape)
        
        x = torch.cat((rgb, cmd, spd),1)
        
        #return self.thr_head(x), self.str_head(x), self.brk_head(x) 
        return self.brk_head(x), self.str_head(x), self.thr_head(x) # Sorting!
