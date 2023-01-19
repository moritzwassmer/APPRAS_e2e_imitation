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
        df_paths_rgb = df_meta_data["dir"] + os.sep + "rgb" + os.sep + df_meta_data["rgb"]
        df_paths_measurements = df_meta_data["dir"] + os.sep + "measurements" + os.sep + df_meta_data["measurements"]
        self.rgb_paths = df_paths_rgb.to_numpy()
        self.measurements_paths = df_paths_measurements.to_numpy()
        #  hotfix for model trainer
        self.y = np.array(["brake", "steer", "throttle"])

    def __len__(self):
        return len(self.rgb_paths)


    def __getitem__(self, idx):

        rgb_np = self.load_rgb(self.rgb_paths[idx])
        measurements = self.load_measurements(self.measurements_paths[idx])
        speed = measurements["speed"]
        command = measurements["command"]
        steer = measurements["steer"]
        throttle = measurements["throttle"]
        brake = measurements["brake"]

        x_sample = {"rgb": rgb_np, "command": command, "speed": speed}
        y_sample = {"brake": brake, "steer": steer, "throttle": throttle}

        return x_sample, y_sample

    def load_rgb(self, path):
        img = cv2.imread(path)
            # reshape to #channels; height; width
        img = img.reshape([3] + list(img.shape)[:-1])
        return img

    def load_measurements(self, path):
        with open(path, 'r') as f:
            measurements = json.load(f)
        return measurements
            