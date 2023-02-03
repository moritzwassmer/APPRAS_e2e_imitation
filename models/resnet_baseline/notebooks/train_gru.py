#%%
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline.utils import create_metadata_df, train_test_split
from data_pipeline.dataset import CARLADataset
from data_pipeline.data_sampler import WeightedSampler
from models.resnet_baseline.baseline_v4 import Baseline_V4
# %%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
path_data = os.path.join( "data", "Dataset_Ege", "test_set", "Town10HD_Scenario10_route12_11_28_18_09_19")

config_xy = {"used_inputs": ["rgb", "measurements"], 
        "used_measurements": ["speed", "steer", "throttle", "brake", "command", "waypoints"],
        "y": ["brake", "steer", "throttle"],
        "seq_len": 1
        }

batch_size = 64
train_dataset = CARLADataset(root_dir=path_data, config=config_xy)
weighted_sampler = WeightedSampler(dataset=train_dataset)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# %%
