from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
import os
import cv2
import json

"""
Inspiration from:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

"""


class CARLADataset(Dataset):

    def __init__(self, root_dir, config, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            used_inputs (list): Contains the folder names of the inputs that shall be used.
            used_measurements (list): Contains the attributes of the measurements that shall be used.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.used_inputs = config["used_inputs"]
        self.used_measurements = config["used_measurements"]
        self.seq_len = config["seq_len"]
        self.df_meta_data = self.__create_metadata_df(root_dir, self.used_inputs)

    def __len__(self):
        return len(self.df_meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_lagged = idx - (self.seq_len - 1)
        sample = dict()
        if not self.__is_idx_valid(idx_lagged, idx):
            return sample

        sample = dict()
        sample["idx"] = torch.arange(idx_lagged, idx + 1)

        for input_idx in range(1, self.df_meta_data.shape[1]):
            input_data_list = []
            for data_point_ix in range(idx_lagged, idx + 1):
                route = self.df_meta_data.iloc[data_point_ix, 0]
                sensor, file_name = self.df_meta_data.columns[input_idx], self.df_meta_data.iloc[data_point_ix, input_idx]
                file_path = os.path.join(self.root_dir, route, sensor, file_name)
                input_data = self.load_data_from_path(file_path)
                input_data_list.append(input_data)
            sample[sensor] = input_data_list
        
        if "measurements" in self.used_inputs:
            sample = self.__select_measurement_attributes(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __is_idx_valid(self, idx_lagged, idx):
        if idx_lagged < 0:
            return False
        route_at_idx = self.df_meta_data.iloc[idx]["route"]
        route_at_idx_lagged = self.df_meta_data.iloc[idx_lagged]["route"]
        if route_at_idx != route_at_idx_lagged:
            return False
        return True


    def __getitem___2(self, idx):
        """
        Old working implementation for seq_len = 0
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()
        sample["idx"] = idx
        route = self.df_meta_data.iloc[idx, 0]
        for j in range(1, self.df_meta_data.shape[1]):
            sensor, file_name = self.df_meta_data.columns[j], self.df_meta_data.iloc[idx, j]
            file_path = os.path.join(self.root_dir, route, sensor, file_name)
            sample[sensor] = self.load_data_from_path(file_path)
        
        if "measurements" in self.used_inputs:
            sample = self.__select_measurement_attributes(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __create_metadata_df(self, root_dir, used_inputs):
        """
        Creates the metadata (i.e. filenames) based on the the data root directory.
        This root directory is supposed to contain folders for individual routes and those folder
        contain folders with the respective sensor types (i.e. lidar) which contain the actual data files.
        This function assumes that for all routes the same measurements/sensors types were recorded!

        Comment: Function could probably be cleaner using os.path.walk()...
        """
        dirs = os.listdir(root_dir)
        route_folders = [dir for dir in dirs if not os.path.isfile(os.path.join(root_dir, dir))]
        sensor_folders = [dir for dir in os.listdir(os.path.join(root_dir, route_folders[0])) if dir in used_inputs]
        columns = ["route"] + sensor_folders
        df_meta = pd.DataFrame(columns=columns)
        for route_folder in route_folders:
            meta_data_route = []
            for sensor_folder in sensor_folders:
                meta_data_route.append(sorted(os.listdir(os.path.join(root_dir, route_folder, sensor_folder))))
            meta_data_route.insert(0, [route_folder]*len(meta_data_route[0]))
            df_meta = pd.concat([df_meta, pd.DataFrame(columns=columns, data=np.transpose(meta_data_route))])        
        return df_meta
    
    def __get_file_path_from_df(self, idx, sensor):
        # TODO: source out functionality from __get_item()
        pass

    def __select_measurement_attributes(self, sample):
        """
        Args:
            sample (dict): The unfiltered sample containing all measurement attributes.
        Return:
            sample_cpy (dict): The filtered sample containing only measurement attributes
            defined in self.used measurements.
        """
        sample_cpy = sample.copy()
        measurements_dict_list = sample_cpy["measurements"]
        for attribute in self.used_measurements:
            measurements_selected = []
            for meas_at_lag_dict in measurements_dict_list:
                measurements_selected.append(meas_at_lag_dict[attribute])
            sample_cpy[attribute] = measurements_selected
        sample_cpy.pop("measurements")
        return sample_cpy

    def __select_measurement_attributes_2(self, sample):
        """
        Args:
            sample (dict): The unfiltered sample containing all measurement attributes.
        Return:
            sample_cpy (dict): The filtered sample containing only measurement attributes 
            defined in self.used measurements.
        """
        sample_cpy = sample.copy()
        measurements = sample_cpy["measurements"]
        for attribute in self.used_measurements:
            sample_cpy[attribute] = measurements[attribute]
        sample_cpy.pop("measurements")
        return sample_cpy


    def save_meta_data(self, path=None):
        """
        Args: 
            path (string): Path where DataFrame containing meta data will be saved.
        """
        if not path:
             path = os.path.join(self.root_dir, "meta_data.csv")
        self.df_meta_data.to_csv(path, index=False)

    def load_data_from_path(self, path):
        """
        Loads the data that can be found in the given path.
        """
        file_format = os.path.splitext(path)[1]
        data = None
        if file_format in [".png", ".jpg"]:
            data = cv2.imread(path)
        elif file_format == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
        elif file_format == ".npy":
            data = np.load(path, allow_pickle=True)
            # discard the weird single number for lidar
            data = data[1]

        return data

                
# create_metadata_csv("myfolder/sample_trainingsdata")



