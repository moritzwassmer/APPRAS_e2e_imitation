from torch.utils.data import Dataset
import datetime
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
import os
import cv2
import json


class CARLADatasetXY(Dataset):

    def __init__(self, root_dir, df_meta_data, config):
        """
        Args:
            root_dir (string): Directory with all the images.
            used_inputs (list): Contains the folder names of the inputs that shall be used.
            used_measurements (list): Contains the attributes of the measurements that shall be used.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.used_inputs = config["used_inputs"]
        self.used_measurements = sorted(config["used_measurements"])
        self.y = sorted(config["y"])
        self.seq_len = config["seq_len"]
        self.df_meta_data = df_meta_data
        self.data_shapes = self.__get_data_shapes()


    def __len__(self):
        return len(self.df_meta_data)

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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_lagged = idx - (self.seq_len - 1)
        x_sample = dict()
        # x_sample["idx"] = torch.arange(idx_lagged, idx + 1)

        for input_idx in range(1, self.df_meta_data.shape[1]):
            if list(self.df_meta_data.columns)[input_idx] == "measurements":
                continue
            shape = [self.seq_len] + self.data_shapes[input_idx - 1]
            data = np.zeros(shape)
            idx_array = 0
            for data_point_idx in range(idx_lagged, idx + 1):
                file_path = self.get_file_path_from_df(input_idx, data_point_idx)
                data_t = self.load_data_from_path(file_path)
                data[idx_array] = data_t
                idx_array += 1
            sensor = self.df_meta_data.columns[input_idx]
            x_sample[sensor] = data
        y_sample = dict()
        if "measurements" in self.used_inputs:
            shape = [len(self.used_measurements), self.seq_len]
            data = np.zeros(shape)
            input_idx = list(self.df_meta_data.columns).index("measurements")
            idx_array = 0
            for data_point_idx in range(idx_lagged, idx + 1):
                file_path = self.get_file_path_from_df(input_idx, data_point_idx)
                data_t = self.load_data_from_path(file_path)
                for meas_idx, meas in enumerate(self.used_measurements):
                    data[meas_idx, idx_array] = data_t[meas]
                idx_array += 1
            for meas_idx, meas in enumerate(self.used_measurements):
                if meas in self.y:
                    y_sample[meas] = data[meas_idx]
                else:
                    x_sample[meas] = data[meas_idx]
        return x_sample, y_sample, idx
            

    def get_file_path_from_df(self, input_idx, data_point_idx):
        route = self.df_meta_data.iloc[data_point_idx, 0]
        sensor, file_name = self.df_meta_data.columns[input_idx], self.df_meta_data.iloc[data_point_idx, input_idx]
        # file_path = os.path.join(self.root_dir, route, sensor, file_name)
        file_path = os.path.join(route, sensor, file_name)
        return file_path


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
        if not os.path.isfile(path):
            print(f"Path is not a file: {path}")
        file_format = os.path.splitext(path)[1]
        data = None
        if file_format in [".png", ".jpg"]:
            data = cv2.imread(path)
            # If rgb
            if data.shape[0] == 160:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                # Reshape to C, H, W
                data = np.transpose(data, (2, 0, 1))
            # LiDAR
            else:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        elif file_format == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
        elif file_format == ".npy":
            data = np.load(path, allow_pickle=True)
            # discard the weird single number for lidar
            data = data[1]
        elif file_format == ".npz":
            with np.load(path, allow_pickle=True) as f:
                data = f["arr_0"]

        return data

    def __get_data_shapes(self):
        path_parts = self.df_meta_data.iloc[0].to_numpy()
        input_types = self.df_meta_data.columns.to_numpy()
        shapes = []
        for i in range(1, len(path_parts)):
            # path = os.path.join(self.root_dir, path_parts[0], input_types[i], path_parts[i])
            path = os.path.join(path_parts[0], input_types[i], path_parts[i])
            data = self.load_data_from_path(path)
            if isinstance(data, np.ndarray):
                shapes.append(list(data.shape))
            else:
                shapes.append(None)
        return shapes
