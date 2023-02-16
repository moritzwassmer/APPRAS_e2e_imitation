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
        # self.y = np.array(["brake", "steer", "throttle"])
        self.y = np.array(["waypoints"])

    def __len__(self):
        return len(self.rgb_paths)


    def __getitem__(self, idx):

        rgb_np = self.load_rgb(self.rgb_paths[idx])
        measurements = self.load_measurements(self.measurements_paths[idx])
        speed = measurements["speed"]
        command = measurements["command"]

        waypoints_global = np.array(measurements["waypoints"])[:4]
        theta = measurements["theta"]
        x_position = measurements["x"]
        y_position = measurements["y"]
        waypoints_local = self.project_global_waypoints_to_local(waypoints_global, x_position, y_position, theta)

        x_sample = {"rgb": rgb_np, "command": command, "speed": np.array([speed])}
        y_sample = {"waypoints": waypoints_local}
        return x_sample, y_sample, idx

    def load_rgb(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # TODO CHANGED TO RGB
        # reshape to #channels; height; width
        #img = img.reshape([3] + list(img.shape)[:-1])
        img = np.transpose(img, (2, 0, 1))
        img = img.astype("float32")
        return img

    def load_measurements(self, path):
        with open(path, 'r') as f:
            measurements = json.load(f)
        return measurements
    
    def project_global_waypoints_to_local(self, waypoints, x, y, theta):
        waypoints = np.asarray(waypoints)
        waypoints = waypoints[:, :2]
        R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
                ])

        local_command_points = np.array([waypoints[:,0]-x, waypoints[:,1]-y])
        local_command_points = R.T.dot(local_command_points)
        return local_command_points.T