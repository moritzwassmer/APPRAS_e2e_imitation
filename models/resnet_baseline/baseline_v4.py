#%%
import torch
import torch.nn as nn
#%%
import torchvision

"""
Input: rgb, speed, navigational command.
Output: break, steer, throttle.
Seq len: 1
Fusion: late fusion (concatenating).
Comment: one dense layers after concatenation. 
"""

class Baseline_V4(nn.Module):
    
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

        self.gru1 =  nn.GRUCell(
            input_size = 10 + 3 + 3, # STATE VECTOR, PAST-WAYPOINT, GOAL-LOCATION
            hidden_size = 10
        )
        self.gru1_dropout = nn.Linear(10, 3)

        self.gru3 =  nn.GRUCell(
            input_size = 10 + 3 + 3, # STATE VECTOR, PAST-WAYPOINT, GOAL-LOCATION
            hidden_size = 10
        )
        self.gru3_dropout = nn.Linear(10, 3)

        self.gru3 =  nn.GRUCell(
            input_size = 10 + 3 + 3, # STATE VECTOR, PAST-WAYPOINT, GOAL-LOCATION
            hidden_size = 10
        )
        self.gru3_dropout = nn.Linear(10, 3)

        self.gru4 =  nn.GRUCell(
            input_size = 10 + 3 + 3, # STATE VECTOR, PAST-WAYPOINT, GOAL-LOCATION
            hidden_size = 10
        )
        self.gru4_dropout = nn.Linear(10, 3)

        self.pos = torch.Tensor((0,0,0))

    # Forward Pass of the Model
    def forward(self, rgb, cmd, spd, goal):
        goal = torch.tensor(goal)
        rgb = self.net(rgb) # BRG
        cmd = self.cmd_input(cmd)
        spd = self.spd_input(spd)
        
        x = torch.cat((rgb, cmd, spd),1)
        x = self.mlp(x)

        x = self.gru1(torch.cat((x, self.pos, goal)))
        wp1 = self.gru1_dropout(x)
        x = self.gru1(torch.cat(x, wp1, goal))
        wp2 = self.gru1_dropout(x)
        x = self.gru1(torch.cat(x, wp2, goal))
        wp3 = self.gru1_dropout(x)
        x = self.gru1(torch.cat(x, wp3, goal))
        wp4 = self.gru1_dropout(x)
        
        #x = self.net.fc(x)
        return wp1, wp2, wp3, wp4