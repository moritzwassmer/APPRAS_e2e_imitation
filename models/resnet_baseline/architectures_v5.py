
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torchvision

"""
Input: rgb, speed, navigational command.
Output: break, steer, throttle.
Seq len: 1
Fusion: late fusion (concatenating).
Comment: one dense layers after concatenation. 
"""


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class Resnet_Baseline_V5(nn.Module):
    
    def __init__(self):
        super().__init__()

        # Number of coordinates for a single waypoint (x,y)=2 OR (x,y,z)=3
        self.num_wyp_coord = 2
        
        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet18(weights=False)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Identity()
        
        # Input Layer fuer cmd, spd
        self.cmd_input = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU() #nn.LeakyReLU() # TODO
        )
        
        self.spd_input = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU() #nn.LeakyReLU() # TODO
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_ftrs+128+128, num_ftrs+8),
            nn.ReLU(), #nn.LeakyReLU()
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(num_ftrs+8, 200),
            nn.ReLU(), #nn.LeakyReLU()
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(200, 100),
            nn.ReLU(), #nn.LeakyReLU()
            #nn.Dropout(p=0.3, inplace=False),
            nn.Linear(100, 30),
            nn.ReLU(), #nn.LeakyReLU()
            #nn.Dropout(p=0.2, inplace=False),
            nn.Linear(30, 10),
            nn.ReLU(), #nn.LeakyReLU()
            #nn.Dropout(p=0.2, inplace=False)
        )

        

        self.gru1 =  nn.GRUCell(
            input_size = 10 + self.num_wyp_coord, # STATE VECTOR, PAST-WAYPOINT, GOAL-LOCATION
            hidden_size = 10
        )
        self.gru1_dropout = nn.Linear(10, self.num_wyp_coord)

        self.gru2 =  nn.GRUCell(
            input_size = 10 + self.num_wyp_coord, # STATE VECTOR, PAST-WAYPOINT, GOAL-LOCATION
            hidden_size = 10
        )
        self.gru2_dropout = nn.Linear(10, self.num_wyp_coord)

        self.gru3 =  nn.GRUCell(
            input_size = 10 + self.num_wyp_coord, # STATE VECTOR, PAST-WAYPOINT, GOAL-LOCATION
            hidden_size = 10
        )
        self.gru3_dropout = nn.Linear(10, self.num_wyp_coord)

        self.gru4 =  nn.GRUCell(
            input_size = 10 + self.num_wyp_coord, # STATE VECTOR, PAST-WAYPOINT, GOAL-LOCATION
            hidden_size = 10
        )
        self.gru4_dropout = nn.Linear(10, self.num_wyp_coord)

        # self.control_turn = PIDController(1.25, .75, .3, 20)
        # self.control_speed = PIDController(5., .5, 1., 20)
        # self.clip_throttle = .25
        # self.brake_ratio = 1.1
        # self.brake_speed = .4
    
    def controller(self, waypoints, speed):
        
        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0

        # BRAKE IF DESIRED SPEED IS SMALL or ACTUAL SPEED IS 10% LARGER THAN DESIRED SPEED
        brake = ((desired_speed < self.brake_speed) or ((speed / desired_speed) > self.brake_ratio))
        
        # CLIP MAXIMUM CHANGE IN SPEED
        delta = np.clip(desired_speed - speed, 0.0, self.clip_throttle)
        throttle = self.control_speed.step(delta)
        throttle = np.clip(throttle, 0.0, self.clip_throttle)

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if (speed < 0.01):
            angle = 0.0  # When we don't move we don't want the angle error to accumulate in the integral
        if brake:
            angle = 0.0
        
        steer = self.control_turn.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        return steer, throttle, brake

    # Forward Pass of the Model
    def forward(self, rgb, cmd, spd):
        rgb = self.net(rgb) # BRG
        cmd = self.cmd_input(cmd)
        spd = self.spd_input(spd)
        
        x_0 = torch.zeros(size=(rgb.shape[0], self.num_wyp_coord), dtype=rgb.dtype, device=rgb.device)
        
        x = torch.cat((rgb, cmd, spd),1)
        x = self.mlp(x)

        x = self.gru1(torch.cat((x, x_0),1))
        wp1 = self.gru1_dropout(x)
        x = self.gru2(torch.cat((x, wp1),1))
        wp2 = self.gru2_dropout(x)
        x = self.gru3(torch.cat((x, wp2),1))
        wp3 = self.gru3_dropout(x)
        x = self.gru4(torch.cat((x, wp3),1))
        wp4 = self.gru4_dropout(x)

        out = torch.stack([wp1,wp2,wp3,wp4],1)

        return out
