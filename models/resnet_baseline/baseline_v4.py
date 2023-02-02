import torch
import torch.nn as nn
import torchvision

"""
Input: rgb, speed, navigational command.
Output: break, steer, throttle.
Seq len: 1
Fusion: late fusion (concatenating).
Comment: one dense layers after concatenation. 
"""

class Baseline_V3(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet18(weights=False)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Identity()
        
        # Input Layer fuer cmd, spd
        self.cmd_input = nn.Sequential(
            nn.Linear(7, 128),
            nn.Tanh() #nn.LeakyReLU() # TODO
        )
        
        self.spd_input = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh() #nn.LeakyReLU() # TODO
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_ftrs+128+128, num_ftrs+8),
            nn.Tanh(), #nn.LeakyReLU()
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(num_ftrs+8, 200),
            nn.Tanh(), #nn.LeakyReLU()
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(200, 100),
            nn.Tanh(), #nn.LeakyReLU()
            #nn.Dropout(p=0.3, inplace=False),
            nn.Linear(100, 30),
            nn.Tanh(), #nn.LeakyReLU()
            #nn.Dropout(p=0.2, inplace=False),
            nn.Linear(30, 10),
            nn.Tanh(), #nn.LeakyReLU()
            #nn.Dropout(p=0.2, inplace=False)
        )

        # REPLACE WITH
        # GATED RECURRENT UNITS
        # OUTPUT WAYPOINTS
        # FEED INTO PID CONTROLER
        # OUTPUT HARD CONTROLS (THROTTLE, BRAKE, STEER)
        
        # Regression Heads for Throttle, Brake and Steering
        self.thr_head = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid() # [0,1] Range Output
        )
        
        self.brk_head = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid() # [0,1] Range Output
        )
        
        self.str_head = nn.Sequential(
            nn.Linear(10, 1),
            nn.Tanh() # [-1,1] Range Output
        )

    # Forward Pass of the Model
    def forward(self, rgb, cmd, spd):
        rgb = self.net(rgb) # BRG
        cmd = self.cmd_input(cmd)
        spd = self.spd_input(spd)
        
        x = torch.cat((rgb, cmd, spd),1)
        x = self.mlp(x)
        
        #x = self.net.fc(x)
        return self.brk_head(x), self.str_head(x), self.thr_head(x) # Sorting!