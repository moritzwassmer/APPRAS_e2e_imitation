#%%
from torch.utils.data import Dataset
import datetime
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
import os
import cv2
import json

"""
This really is the smallest, most-specialized Dataset possible for our scenario and still it isn't
faster than the much bigger more specialized version.
"""

class CARLADatasetXYOpt(Dataset):

    def __init__(self, df_meta_data):
        """
        Args:
            root_dir (string): Directory with all the images.
            used_inputs (list): Contains the folder names of the inputs that shall be used.
            used_measurements (list): Contains the attributes of the measurements that shall be used.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df_meta_data = df_meta_data
        df_paths_rgb = df_meta_data["dir"] + os.sep + "rgb" + os.sep + df_meta_data["rgb"]
        df_paths_measurements = df_meta_data["dir"] + os.sep + "measurements" + os.sep + df_meta_data["measurements"]
        self.rgb_paths = df_paths_rgb.to_numpy()
        self.measurements_paths = df_paths_measurements.to_numpy()
        #  hotfix for model trainer
        # self.y = np.array(["brake", "steer", "throttle"])
        self.y = np.array(["waypoints"])

    def __len__(self):
        return len(self.rgb_paths)


    def __getitem__(self, idx):

        rgb_np = self.load_rgb(self.rgb_paths[idx]).astype(float)
        measurements = self.load_measurements(self.measurements_paths[idx])
        speed = measurements["speed"]
        speed = np.array([speed])
        command = measurements["command"]
        steer = measurements["steer"]
        throttle = measurements["throttle"]
        brake = measurements["brake"]
        waypoints = measurements["waypoints"]
        waypoints = waypoints[:4]

        x_sample = {"rgb": rgb_np, "command": command, "speed": speed}
        # y_sample = {"brake": brake, "steer": steer, "throttle": throttle}
        y_sample = {"waypoints": np.array(waypoints)}

        return x_sample, y_sample, idx

    def load_rgb(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # TODO CHANGED TO RGB
            # reshape to #channels; height; width
        img = img.reshape([3] + list(img.shape)[:-1])
        return img

    def load_measurements(self, path):
        with open(path, 'r') as f:
            measurements = json.load(f)
        return measurements

    def get_statistics(self):
        df_meta_data = self.df_meta_data
        df_meta_data_full_paths = df_meta_data[df_meta_data.columns[1:]].apply(lambda x: df_meta_data["dir"] + os.sep + x.name + os.sep + x)
        df_meta_data_sizes = df_meta_data_full_paths.applymap(lambda path: os.path.getsize(path))
        df_stats = (df_meta_data_sizes.sum() / 10**9).round(2).to_frame().T
        df_stats.columns = df_stats.columns + "_in_GB" 
        # df_stats["time_hours"] = len(df_meta_data) / (2 * 60 * 60)
        df_stats["driving_time"] = str(datetime.timedelta(seconds=int(len(df_meta_data) / 2)))
        df_stats["%_of_entire_data"] = round((len(df_meta_data) / 258866 * 100), 2)
        return df_stats
            
# %%
