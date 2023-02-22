import torch.nn as nn
import torchvision
import torch

from torchvision.models import resnet18, ResNet18_Weights


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


def mlp(neurons_in, neurons_out, neurons_hidden):
    return (nn.Sequential(
        nn.Linear(neurons_in, neurons_hidden),
        nn.Sigmoid(),
        nn.Linear(neurons_hidden, neurons_hidden),
        nn.Sigmoid(),
        nn.Linear(neurons_hidden, neurons_out),
        nn.Sigmoid()
    ))


def to_cuda_if_possible(data, device): # TODO HAS TO BE LIKE THAT FOR INFERENCE
    return data.to(device) if device else data


class Resnet_Baseline_V4_Shuffle(nn.Module):

    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet34(weights=True)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False)
            #nn.Identity()
        )

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
        self.shared = nn.Sequential(
            nn.Identity()

        )

        self.branches_mlp = nn.ModuleList()
        self.branches_brake = nn.ModuleList()
        self.branches_steer = nn.ModuleList()
        self.branches_throttle = nn.ModuleList()

        # Create Branches for all Commands
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

    def forward(self, rgb, nav, spd):

        # Pass until shared layer reached
        rgb = self.net(rgb)  # BRG
        spd = self.spd_input(spd)
        x = torch.cat((rgb, spd), 1)
        x = self.shared(x)

        # Move Data to respective Branch depending on Command
        index = to_cuda_if_possible(torch.arange(len(nav[:, 0])), self.device)
        cmd = to_cuda_if_possible(torch.where(nav)[1], self.device)
        index = to_cuda_if_possible(torch.unsqueeze(index, 1), self.device)
        cmd = to_cuda_if_possible(torch.unsqueeze(cmd, 1), self.device)
        # print(index.shape)
        # print(cmd.shape)
        mapping = to_cuda_if_possible(torch.cat((index, cmd), axis=1), self.device)
        # print(mapping)
        # mapping_sorted = to_cuda_if_possible(torch.sort(mapping,0).values)
        classes = to_cuda_if_possible(torch.unique(cmd), self.device)

        # Mapping which samples have to go to which branch
        mapping_list = []
        for idx, i in enumerate(classes):
            mapping = to_cuda_if_possible(torch.where(cmd == i)[0], self.device)
            mapping_list.append(mapping)
        # print(mapping_list)

        # Move samples to the branches and predict
        preds = []
        for i in range(len(classes)):
            # print("predict on "+str(classes[i]))
            mlp_output = self.branches_mlp[classes[i]](x)
            brake = self.branches_brake[classes[i]](mlp_output[mapping_list[i]])
            steer = self.branches_steer[classes[i]](mlp_output[mapping_list[i]])  #
            throttle = self.branches_throttle[classes[i]](mlp_output[mapping_list[i]])

            y_hat = torch.cat((brake, steer, throttle), axis=1)
            # print(classes[i])
            # print(y_hat.shape)
            preds.append(y_hat)
            # print(preds)

        # Map the Predictions with the original positions
        map_pred_list = []
        for i in range(len(classes)):
            mapping_list[i] = torch.unsqueeze(mapping_list[i], 1)
            concatenated = torch.cat((mapping_list[i], preds[i]), axis=1)
            map_pred_list.append(concatenated)
            # print(concatenated)
        stacked = torch.cat(map_pred_list, 0)
        # print(stacked)

        # Sort back to original positions
        # sorted_out = torch.sort(stacked,0).values
        sorted_out = stacked[stacked[:, 0].sort()[1]]  # a[a[:, 0].sort()[1]]

        # print(sorted_out)

        # Extract Policy
        brake = torch.unsqueeze(sorted_out[:, 1], 1)
        steer = torch.unsqueeze(sorted_out[:, 2], 1)
        throttle = torch.unsqueeze(sorted_out[:, 3], 1)

        return brake, steer, throttle

class Resnet_Baseline_V4_No_Shuffle(nn.Module):

    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet34(weights=True)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Sequential(
            nn.Dropout(p=0, inplace=False)
            # nn.Identity()
        )

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
        self.shared = nn.Sequential(
            nn.Identity()

        )

        self.branches_mlp = nn.ModuleList()
        self.branches_brake = nn.ModuleList()
        self.branches_steer = nn.ModuleList()
        self.branches_throttle = nn.ModuleList()

        # Create Branches for all Commands
        for i in range(0, 7): # TODO Branch 0 not used, should be adjusted later on
            mlp_branch = mlp(num_ftrs + 128, 256, 256)
            brk = brake_head(256)
            thr = throttle_head(256)
            steer = steer_head(256)

            self.branches_mlp.append(mlp_branch)
            self.branches_brake.append(brk)
            self.branches_steer.append(steer)
            self.branches_throttle.append(thr)

            # Forward Pass of the Model

    def forward(self, rgb, spd, batch_size):

        # Pass until shared layer reached
        rgb = self.net(rgb)  # BRG
        spd = self.spd_input(spd)
        x = torch.cat((rgb, spd), 1)
        x = self.shared(x)

        preds = []
        for i in range(1,6):
            mlp_output = self.branches_mlp[i](x)
            brake = self.branches_brake[i](mlp_output[i])
            steer = self.branches_steer[i](mlp_output[i])  #
            throttle = self.branches_throttle[i](mlp_output[i])
            y_hat = torch.cat((brake, steer, throttle), axis=1)
            preds.append(y_hat)


        stacked = torch.cat(preds, 0)

        # Extract Policy
        brake = torch.unsqueeze(stacked[:, 1], 1)
        steer = torch.unsqueeze(stacked[:, 2], 1)
        throttle = torch.unsqueeze(stacked[:, 3], 1)

        return brake, steer, throttle


class Resnet_Baseline_V4_No_Shuffle_Drop_MLP(nn.Module):

    def mlp(self, neurons_in, neurons_out, neurons_hidden):
        return (nn.Sequential(
            nn.Linear(neurons_in, neurons_hidden),
            nn.Sigmoid(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(neurons_hidden, neurons_hidden),
            nn.Sigmoid(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(neurons_hidden, neurons_out),
            nn.Sigmoid(),
            nn.Dropout(p=0.3, inplace=False),
        ))

    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet34(weights=True)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=False)
            nn.Identity()
            # nn.Linear(512, 512),
            # nn.Tanh(),
        )

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
        self.shared = nn.Sequential(
            nn.Identity()

        )

        self.branches_mlp = nn.ModuleList()
        self.branches_brake = nn.ModuleList()
        self.branches_steer = nn.ModuleList()
        self.branches_throttle = nn.ModuleList()

        # Create Branches for all Commands
        for i in range(0, 7):
            mlp_branch = self.mlp(num_ftrs + 128, 256, 256)
            brk = brake_head(256)
            thr = throttle_head(256)
            steer = steer_head(256)

            self.branches_mlp.append(mlp_branch)
            self.branches_brake.append(brk)
            self.branches_steer.append(steer)
            self.branches_throttle.append(thr)

            # Forward Pass of the Model

    def forward(self, rgb, nav, spd):

        # Pass until shared layer reached
        rgb = self.net(rgb)  # BRG
        spd = self.spd_input(spd)
        x = torch.cat((rgb, spd), 1)
        x = self.shared(x)
        # print(rgb.shape)

        # Move Data to respective Branch depending on Command
        index = to_cuda_if_possible(torch.arange(len(nav[:, 0])), self.device)
        cmd = to_cuda_if_possible(torch.where(nav)[1], self.device)
        index = to_cuda_if_possible(torch.unsqueeze(index, 1), self.device)
        cmd = to_cuda_if_possible(torch.unsqueeze(cmd, 1), self.device)
        # mapping = to_cuda_if_possible(torch.cat((index,cmd),axis=1),self.device)
        classes = to_cuda_if_possible(torch.unique(cmd), self.device)

        # indices of which samples have to go to which branch
        mapping_list = []
        for idx, i in enumerate(classes):  # [2,3,4]
            mapping = to_cuda_if_possible(torch.where(cmd == i)[0], self.device)
            mapping_list.append(mapping)

        # Move samples to the branches and predict
        preds = []
        for i in range(len(classes)):
            mlp_output = self.branches_mlp[classes[i]](x)
            brake = self.branches_brake[classes[i]](mlp_output[mapping_list[i]])
            steer = self.branches_steer[classes[i]](mlp_output[mapping_list[i]])  #
            throttle = self.branches_throttle[classes[i]](mlp_output[mapping_list[i]])

            y_hat = torch.cat((brake, steer, throttle), axis=1)
            preds.append(y_hat)

        # Map the Predictions with the original positions
        map_pred_list = []
        for i in range(len(classes)):  # 012
            mapping_list[i] = torch.unsqueeze(mapping_list[i], 1)
            concatenated = torch.cat((mapping_list[i], preds[i]), axis=1)
            map_pred_list.append(concatenated)
        stacked = torch.cat(map_pred_list, 0)

        # Sort back to original positions
        sorted_out = stacked[stacked[:, 0].sort()[1]]  # a[a[:, 0].sort()[1]]

        # Extract Policy
        brake = torch.unsqueeze(sorted_out[:, 1], 1)
        steer = torch.unsqueeze(sorted_out[:, 2], 1)
        throttle = torch.unsqueeze(sorted_out[:, 3], 1)

        return brake, steer, throttle

class Long_Run(nn.Module):

    def mlp(self, neurons_in, neurons_out, neurons_hidden):
        return (nn.Sequential(
            nn.Linear(neurons_in, neurons_hidden),
            nn.Sigmoid(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(neurons_hidden, neurons_hidden),
            nn.Sigmoid(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(neurons_hidden, neurons_out),
            nn.Sigmoid()
        ))

    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=False)
            nn.Identity()
        )

        self.spd_input = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 128),
            nn.Tanh()  # nn.LeakyReLU() # TODO
            # nn.Dropout(p=0.5, inplace=False)
        )

        # shared MLP
        self.shared = nn.Sequential(
            nn.Identity()

        )

        self.branches_mlp = nn.ModuleList()
        self.branches_brake = nn.ModuleList()
        self.branches_steer = nn.ModuleList()
        self.branches_throttle = nn.ModuleList()

        # Create Branches for all Commands
        for i in range(0, 7):
            mlp_branch = self.mlp(num_ftrs + 128, 256, 256)
            brk = brake_head(256)
            thr = throttle_head(256)
            steer = steer_head(256)

            self.branches_mlp.append(mlp_branch)
            self.branches_brake.append(brk)
            self.branches_steer.append(steer)
            self.branches_throttle.append(thr)

            # Forward Pass of the Model

    def forward(self, rgb, nav, spd):

        # Pass until shared layer reached
        rgb = self.net(rgb)  # BRG
        spd = self.spd_input(spd)
        x = torch.cat((rgb, spd), 1)
        x = self.shared(x)
        # print(rgb.shape)

        # Move Data to respective Branch depending on Command
        index = to_cuda_if_possible(torch.arange(len(nav[:, 0])), self.device)
        cmd = to_cuda_if_possible(torch.where(nav)[1], self.device)
        index = to_cuda_if_possible(torch.unsqueeze(index, 1), self.device)
        cmd = to_cuda_if_possible(torch.unsqueeze(cmd, 1), self.device)
        # mapping = to_cuda_if_possible(torch.cat((index,cmd),axis=1),self.device)
        classes = to_cuda_if_possible(torch.unique(cmd), self.device)

        # indices of which samples have to go to which branch
        mapping_list = []
        for idx, i in enumerate(classes):  # [2,3,4]
            mapping = to_cuda_if_possible(torch.where(cmd == i)[0], self.device)
            mapping_list.append(mapping)

        # Move samples to the branches and predict
        preds = []
        for i in range(len(classes)):
            mlp_output = self.branches_mlp[classes[i]](x)
            brake = self.branches_brake[classes[i]](mlp_output[mapping_list[i]])
            steer = self.branches_steer[classes[i]](mlp_output[mapping_list[i]])  #
            throttle = self.branches_throttle[classes[i]](mlp_output[mapping_list[i]])

            y_hat = torch.cat((brake, steer, throttle), axis=1)
            preds.append(y_hat)

        # Map the Predictions with the original positions
        map_pred_list = []
        for i in range(len(classes)):  # 012
            mapping_list[i] = torch.unsqueeze(mapping_list[i], 1)
            concatenated = torch.cat((mapping_list[i], preds[i]), axis=1)
            map_pred_list.append(concatenated)
        stacked = torch.cat(map_pred_list, 0)

        # Sort back to original positions
        sorted_out = stacked[stacked[:, 0].sort()[1]]  # a[a[:, 0].sort()[1]]

        # Extract Policy
        brake = torch.unsqueeze(sorted_out[:, 1], 1)
        steer = torch.unsqueeze(sorted_out[:, 2], 1)
        throttle = torch.unsqueeze(sorted_out[:, 3], 1)

        return brake, steer, throttle

from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights

class Long_Run_2(nn.Module):

    def mlp(self, neurons_in, neurons_out, neurons_hidden):
        return (nn.Sequential(
            nn.Linear(neurons_in, neurons_hidden),
            nn.ReLU(),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(neurons_hidden, neurons_hidden),
            nn.ReLU(),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(neurons_hidden, neurons_out),
            nn.ReLU()
        ))

    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False)
            #nn.Identity()
        )

        self.spd_input = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 128),
            nn.ReLU()  # nn.LeakyReLU() # TODO
            # nn.Dropout(p=0.5, inplace=False)
        )

        # shared MLP
        self.shared = nn.Sequential(
            nn.Identity()
        )

        self.branches_mlp = nn.ModuleList()
        self.branches_brake = nn.ModuleList()
        self.branches_steer = nn.ModuleList()
        self.branches_throttle = nn.ModuleList()

        # Create Branches for all Commands
        for i in range(0, 7):
            mlp_branch = self.mlp(num_ftrs + 128, 256, 256)
            brk = brake_head(256)
            thr = throttle_head(256)
            steer = steer_head(256)

            self.branches_mlp.append(mlp_branch)
            self.branches_brake.append(brk)
            self.branches_steer.append(steer)
            self.branches_throttle.append(thr)

            # Forward Pass of the Model

    def forward(self, rgb, nav, spd):

        # Pass until shared layer reached
        rgb = self.net(rgb)  # BRG
        spd = self.spd_input(spd)
        x = torch.cat((rgb, spd), 1)
        x = self.shared(x)
        # print(rgb.shape)

        # Move Data to respective Branch depending on Command
        index = to_cuda_if_possible(torch.arange(len(nav[:, 0])), self.device)
        cmd = to_cuda_if_possible(torch.where(nav)[1], self.device)
        index = to_cuda_if_possible(torch.unsqueeze(index, 1), self.device)
        cmd = to_cuda_if_possible(torch.unsqueeze(cmd, 1), self.device)
        # mapping = to_cuda_if_possible(torch.cat((index,cmd),axis=1),self.device)
        classes = to_cuda_if_possible(torch.unique(cmd), self.device)

        # indices of which samples have to go to which branch
        mapping_list = []
        for idx, i in enumerate(classes):  # [2,3,4]
            mapping = to_cuda_if_possible(torch.where(cmd == i)[0], self.device)
            mapping_list.append(mapping)

        # Move samples to the branches and predict
        preds = []
        for i in range(len(classes)):
            mlp_output = self.branches_mlp[classes[i]](x)
            brake = self.branches_brake[classes[i]](mlp_output[mapping_list[i]])
            steer = self.branches_steer[classes[i]](mlp_output[mapping_list[i]])  #
            throttle = self.branches_throttle[classes[i]](mlp_output[mapping_list[i]])

            y_hat = torch.cat((brake, steer, throttle), axis=1)
            preds.append(y_hat)

        # Map the Predictions with the original positions
        map_pred_list = []
        for i in range(len(classes)):  # 012
            mapping_list[i] = torch.unsqueeze(mapping_list[i], 1)
            concatenated = torch.cat((mapping_list[i], preds[i]), axis=1)
            map_pred_list.append(concatenated)
        stacked = torch.cat(map_pred_list, 0)

        # Sort back to original positions
        sorted_out = stacked[stacked[:, 0].sort()[1]]

        # Extract Policy
        brake = torch.unsqueeze(sorted_out[:, 1], 1)
        steer = torch.unsqueeze(sorted_out[:, 2], 1)
        throttle = torch.unsqueeze(sorted_out[:, 3], 1)

        return brake, steer, throttle

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

def to_cuda_if_possible(data, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')): # TODO HAS TO BE LIKE THAT FOR INFERENCE
    return data.to(device) if device else data


class MyResnet(nn.Module):

    def mlp(self, neurons_in, neurons_out, neurons_hidden):
        return (nn.Sequential(
            nn.Linear(neurons_in, neurons_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(neurons_hidden, neurons_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(neurons_hidden, neurons_out),
            nn.ReLU()
        ))

    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Sequential(
            nn.Dropout(p=0.18, inplace=False)
            #nn.Identity()
        )

        self.spd_input = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(128, 128),
            nn.ReLU(),  # nn.LeakyReLU() # TODO
            nn.Dropout(p=0.1, inplace=False)
        )

        # shared MLP
        self.shared = nn.Sequential(
            self.mlp(num_ftrs + 128, num_ftrs + 128, num_ftrs + 128)
        )

        self.branches_mlp = nn.ModuleList()
        self.branches_brake = nn.ModuleList()
        self.branches_steer = nn.ModuleList()
        self.branches_throttle = nn.ModuleList()

        # Create Branches for all Commands
        for i in range(0, 7):
            mlp_branch = self.mlp(num_ftrs + 128, 256, 256)
            brk = brake_head(256)
            thr = throttle_head(256)
            steer = steer_head(256)

            self.branches_mlp.append(mlp_branch)
            self.branches_brake.append(brk)
            self.branches_steer.append(steer)
            self.branches_throttle.append(thr)

            # Forward Pass of the Model

    def forward(self, rgb, nav, spd):

        # Pass until shared layer reached
        rgb = self.net(rgb)  # BRG
        spd = self.spd_input(spd)
        x = torch.cat((rgb, spd), 1)
        x = self.shared(x)
        # print(rgb.shape)

        # Move Data to respective Branch depending on Command
        index = to_cuda_if_possible(torch.arange(len(nav[:, 0])), self.device)
        cmd = to_cuda_if_possible(torch.where(nav)[1], self.device)
        index = to_cuda_if_possible(torch.unsqueeze(index, 1), self.device)
        cmd = to_cuda_if_possible(torch.unsqueeze(cmd, 1), self.device)
        # mapping = to_cuda_if_possible(torch.cat((index,cmd),axis=1),self.device)
        classes = to_cuda_if_possible(torch.unique(cmd), self.device)

        # indices of which samples have to go to which branch
        mapping_list = []
        for idx, i in enumerate(classes):  # [2,3,4]
            mapping = to_cuda_if_possible(torch.where(cmd == i)[0], self.device)
            mapping_list.append(mapping)

        # Move samples to the branches and predict
        preds = []
        for i in range(len(classes)):
            mlp_output = self.branches_mlp[classes[i]](x)
            brake = self.branches_brake[classes[i]](mlp_output[mapping_list[i]])
            steer = self.branches_steer[classes[i]](mlp_output[mapping_list[i]])  #
            throttle = self.branches_throttle[classes[i]](mlp_output[mapping_list[i]])

            y_hat = torch.cat((brake, steer, throttle), axis=1)
            preds.append(y_hat)

        # Map the Predictions with the original positions
        map_pred_list = []
        for i in range(len(classes)):  # 012
            mapping_list[i] = torch.unsqueeze(mapping_list[i], 1)
            concatenated = torch.cat((mapping_list[i], preds[i]), axis=1)
            map_pred_list.append(concatenated)
        stacked = torch.cat(map_pred_list, 0)

        # Sort back to original positions
        sorted_out = stacked[stacked[:, 0].sort()[1]]

        # Extract Policy
        brake = torch.unsqueeze(sorted_out[:, 1], 1)
        steer = torch.unsqueeze(sorted_out[:, 2], 1)
        throttle = torch.unsqueeze(sorted_out[:, 3], 1)

        return brake, steer, throttle