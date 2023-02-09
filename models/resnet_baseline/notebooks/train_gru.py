#%%
import sys
sys.path.append("")
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline.data_preprocessing import preprocessing
from data_pipeline.utils import create_metadata_df, train_test_split
from data_pipeline.dataset_xy import CARLADatasetXY
from data_pipeline.dataset_xy_opt import CARLADatasetXYOpt
from data_pipeline.data_sampler import WeightedSampler
from models.resnet_baseline.baseline_v4 import Baseline_V4
from models.model_trainer import ModelTrainer
# %%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
path_data = os.path.join( "data", "Dataset_Ege", "test_set", "Town10HD_Scenario10_route12_11_28_18_09_19")

config_xy = {"used_inputs": ["rgb", "measurements"], 
        "used_measurements": ["speed", "steer", "throttle", "brake", "command"],
        "y": ["brake", "steer", "throttle"],
        "seq_len": 1
        }

df_meta_data = create_metadata_df(path_data, config_xy["used_inputs"])


batch_size = 32
# train_dataset = CARLADatasetXY(root_dir=path_data, df_meta_data=df_meta_data, config=config_xy)
# weighted_sampler = WeightedSampler(dataset=train_dataset)
train_dataset = CARLADatasetXYOpt(df_meta_data)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# %%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
print(device)
model = Baseline_V4()
model.to(device)
# Loss and Optimizer
criterion = nn.L1Loss() # Easy to interpret #nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 10
print_every = 200
valid_loss_min = np.Inf
val_loss = []
train_loss = []
total_step = len(train_dataloader)

validate = True
loss_fns_dict = {"waypoints": nn.L1Loss(reduction='none')}
loss_fn_weights = {"waypoints": 1.}

model_trainer = ModelTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
    loss_fns=loss_fns_dict,
    loss_fn_weights=loss_fn_weights,
    sample_weights=None,
    n_epochs=10,
    dataloader_train=train_dataloader,
    dataloader_test=test_dataloader,
    preprocessing=preprocessing,
    upload_tensorboard=True
    )
model_trainer.run()

# %%
