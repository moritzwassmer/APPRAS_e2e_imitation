import torch.nn as nn
import torchvision
import torch

def mlp(neurons_in, neurons_out, neurons_hidden):
    """ Creates a MLP layer with Relu activation

    Args:
        neurons_in: dimensionality of Input data
        neurons_out: dimensionality of Output layer
        neurons_hidden: dimensionality of Hidden layer

    Returns:
        Sequential Module with linear layer with Relu activations
    """
    return (nn.Sequential(
        nn.Linear(neurons_in, neurons_hidden),
        nn.Tanh(),
        nn.Linear(neurons_hidden, neurons_hidden),
        nn.Tanh(),
        nn.Linear(neurons_hidden, neurons_out),
        nn.Tanh()
    ))

def steer_head(neurons_in):  # [-1,1] Range Output
    """ Creates a linear output layer with tanh activation

    Args:
        neurons_in: dimensionality of Input data

    Returns:
        Sequential Module with linear layer with tanh activation
    """
    str_head = nn.Sequential(
        nn.Linear(neurons_in, 1),
        nn.Tanh())
    return str_head


def throttle_head(neurons_in):  # [0,1] Range Output
    """ Creates a linear output layer with sigmoid activation

    Args:
        neurons_in: dimensionality of Input data

    Returns:
        Sequential Module with linear layer with sigmoid activation
    """
    thr_head = nn.Sequential(
        nn.Linear(neurons_in, 1),
        nn.Sigmoid())
    return thr_head


def brake_head(neurons_in):  # [0,1] Range Output
    """ Creates a linear output layer with sigmoid activation

    Args:
        neurons_in: dimensionality of Input data

    Returns:
        Sequential Module with linear layer with sigmoid activation
    """
    brk_head = nn.Sequential(
        nn.Linear(neurons_in, 1),
        nn.Sigmoid())
    return brk_head

def to_cuda_if_possible(data, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')): #
    """ Moves data to cuda

    It does assume all navigational commands in batch to be equally present in the dataset and sorted.
    This Model did yield worse performances in our tests which is why we chose V3 for Evaluation

    Args:
        data: Torch Tensor
        device: torch device (GPU, MPS, CPU)

    Returns:
        Cuda compatible torch tensor
    """
    return data.to(device) if device else data

class Resnet_Baseline_V4(nn.Module): # Initial Version. Is not able to handle a dataloader that uses shuffling. Needs to use CommandSampler

    """ Class implementing the branched Architecture of CIL paper https://arxiv.org/abs/1904.08980 without speed regularization

    It does assume all navigational commands in batch to be equally present in the dataset and sorted.
    This Model did yield worse performances in our tests which is why we chose V3 for Evaluation

    Attributes:
        net: vision backbone (ResNet)
        spd_input: speed input layer
        mlp: MLP Module
        branches_mlp: ModuleList for MLP Modules for respective branch
        branches_brake: ModuleList for brake Modules (OutputLayer) for respective branch
        branches_steer: ModuleList for steel Modules (OutputLayer) for respective branch
        branches_throttle: ModuleList for throttle Modules (OutputLayer) for respective branch
    """

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

        self.spd_input = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Identity()

        )

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
    def forward(self, rgb, cmd, spd):
        rgb = self.net(rgb)  # BRG
        spd = self.spd_input(spd)

        x = torch.cat((rgb, spd), 1)
        x = self.mlp(x)

        idx = torch.where(cmd[0])[0]
        x = self.branches_mlp[idx](x)

        brake = self.branches_brake[idx](x)
        steer = self.branches_steer[idx](x)
        throttle = self.branches_throttle[idx](x)

        return brake, steer, throttle

class Resnet_Baseline_V4_Shuffle(nn.Module): # Able to use any Dataloader.

    """ Class implementing the branched Architecture of CIL paper https://arxiv.org/abs/1904.08980 without speed regularization

    Automatically splits up the batch and sends samples according to their command value to the corresponding branch.
    That enables use of shuffling of training data and therefore is not bound to a specific Sampling Strategy.
    This Model did yield worse performances in our tests which is why we chose V3 for Evaluation

    Attributes:
        net: vision backbone (ResNet)
        spd_input: speed input layer
        mlp: MLP Module
        branches_mlp: ModuleList for MLP Modules for respective branch
        branches_brake: ModuleList for brake Modules (OutputLayer) for respective branch
        branches_steer: ModuleList for steel Modules (OutputLayer) for respective branch
        branches_throttle: ModuleList for throttle Modules (OutputLayer) for respective branch
    """


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
            nn.Tanh()
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
        cmd = to_cuda_if_possible(torch.where(nav)[1], self.device)
        cmd = to_cuda_if_possible(torch.unsqueeze(cmd, 1), self.device)

        classes = to_cuda_if_possible(torch.unique(cmd), self.device)

        # Mapping which samples have to go to which branch
        mapping_list = []
        for idx, i in enumerate(classes):
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
        for i in range(len(classes)):
            mapping_list[i] = torch.unsqueeze(mapping_list[i], 1)
            concatenated = torch.cat((mapping_list[i], preds[i]), axis=1)
            map_pred_list.append(concatenated)
        stacked = torch.cat(map_pred_list, 0)

        sorted_out = stacked[stacked[:, 0].sort()[1]]  # a[a[:, 0].sort()[1]]

        # Extract Policy
        brake = torch.unsqueeze(sorted_out[:, 1], 1)
        steer = torch.unsqueeze(sorted_out[:, 2], 1)
        throttle = torch.unsqueeze(sorted_out[:, 3], 1)

        return brake, steer, throttle


from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights

class Resnet_Baseline_V4_Shuffle_2(nn.Module): # Old Name: Long_Run_2

    """ Class implementing the branched Architecture of CIL paper https://arxiv.org/abs/1904.08980 without speed regularization

    Automatically splits up the batch and sends samples according to their command value to the corresponding branch.
    That enables use of shuffling of training data and therefore is not bound to a specific Sampling Strategy.
    This Model did yield worse performances in our tests which is why we chose V3 for Evaluation
    Replaced Hidden activations from tanh -> Relu

    Attributes:
        net: vision backbone (ResNet)
        spd_input: speed input layer
        mlp: MLP Module
        branches_mlp: ModuleList for MLP Modules for respective branch
        branches_brake: ModuleList for brake Modules (OutputLayer) for respective branch
        branches_steer: ModuleList for steel Modules (OutputLayer) for respective branch
        branches_throttle: ModuleList for throttle Modules (OutputLayer) for respective branch
    """

    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ResNet Architecture with pretrained weights, also bigger resnets available
        self.net = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        num_ftrs = self.net.fc.in_features

        # Top layer of ResNet which you can modify. We choose Identity to use it as Input for all the heads
        self.net.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False)
        )

        self.spd_input = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
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


        # Move Data to respective Branch depending on Command
        cmd = to_cuda_if_possible(torch.where(nav)[1], self.device)
        cmd = to_cuda_if_possible(torch.unsqueeze(cmd, 1), self.device)
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

    def mlp(neurons_in, neurons_out, neurons_hidden):
        """ Creates a MLP layer with Relu activation

        Args:
            neurons_in: dimensionality of Input data
            neurons_out: dimensionality of Output layer
            neurons_hidden: dimensionality of Hidden layer

        Returns:
            Sequential Module with linear layer with Relu activations
        """
        return (nn.Sequential(
            nn.Linear(neurons_in, neurons_hidden),
            nn.ReLU(),
            nn.Linear(neurons_hidden, neurons_hidden),
            nn.ReLU(),
            nn.Linear(neurons_hidden, neurons_out),
            nn.ReLU()
        ))
