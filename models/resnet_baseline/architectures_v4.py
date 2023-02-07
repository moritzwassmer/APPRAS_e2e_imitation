import torch.nn as nn
import torchvision
import torch


def mlp(neurons_in, neurons_out, neurons_hidden):
    return (nn.Sequential(
        nn.Linear(neurons_in, neurons_hidden),
        nn.Sigmoid(),
        nn.Linear(neurons_hidden, neurons_hidden),
        nn.Sigmoid(),
        nn.Linear(neurons_hidden, neurons_out),
        nn.Sigmoid()
    ))


def steer_head(neurons_in):  # [-1,1] Range Output
    str_head = nn.Sequential(
        nn.Linear(neurons_in, 1),
        nn.Tanh())
    return str_head


def throttle_head(neurons_in):  # [0,1] Range Output
    thr_head = nn.Sequential(
        nn.Linear(neurons_in, 1),
        nn.Sigmoid())
    return thr_head


def brake_head(neurons_in):  # [0,1] Range Output
    brk_head = nn.Sequential(
        nn.Linear(neurons_in, 1),
        nn.Sigmoid())
    return brk_head

class Resnet_Baseline_V4(nn.Module):

    def __init__(self):
        super().__init__()

        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet34(weights=True)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False)
            # nn.Identity()
        )

        """
        # Input Layer fuer cmd, spd
        self.cmd_input = nn.Sequential(
            nn.Linear(7, 128),
            nn.Tanh() #nn.LeakyReLU() # TODO
            #nn.Dropout(p=0.5, inplace=False)
        )
        """

        self.spd_input = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()  # nn.LeakyReLU() # TODO
            # nn.Dropout(p=0.5, inplace=False)
        )

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Identity()

        )
        """
            nn.Linear(num_ftrs+128, num_ftrs+128),
            nn.Tanh(), #nn.LeakyReLU()
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(num_ftrs+128, num_ftrs+128),
            nn.Tanh(), #nn.LeakyReLU()
            nn.Dropout(p=0.1, inplace=False)
        """
        """
        self.W_list =  nn.ModuleList()

        for i  in range(relation_num):
        self.W_list.append(nn.Linear(self.dim, self.dim))
        """
        self.branches_mlp = nn.ModuleList()
        self.branches_brake = nn.ModuleList()
        self.branches_steer = nn.ModuleList()
        self.branches_throttle = nn.ModuleList()

        for i in range(0, 7):
            mlp_branch = mlp(num_ftrs + 128, 256, 256)
            brk = brake_head(256)
            thr = throttle_head(256)
            steer = steer_head(256)

            self.branches_mlp.append(mlp_branch)
            self.branches_brake.append(brk)
            self.branches_steer.append(steer)
            self.branches_throttle.append(thr)

    # Forward Pass of the Model

    """
    def forward(cond, x):
    x=self.layer1(x)
    if cond:
        x=self.relu(x)
    ...
    return x
    """

    def forward(self, rgb, cmd, spd):
        rgb = self.net(rgb)  # BRG
        # rgb = self.net.fc(rgb)
        # cmd = self.cmd_input(cmd)
        spd = self.spd_input(spd)

        x = torch.cat((rgb, spd), 1)
        x = self.mlp(x)

        idx = torch.where(cmd[0])[0]
        # print(idx)
        x = self.branches_mlp[idx](x)

        brake = self.branches_brake[idx](x)
        steer = self.branches_steer[idx](x)
        throttle = self.branches_throttle[idx](x)

        return brake, steer, throttle
