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

    def __init__(self, root_dir, config):
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
        self.used_measurements = config["used_measurements"]
        self.seq_len = config["seq_len"]
        self.df_meta_data = self.__create_metadata_df(root_dir, self.used_inputs)
        self.data_shapes = self.__get_data_shapes()

    def __len__(self):
        return len(self.df_meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_lagged = idx - (self.seq_len - 1)
        sample = dict()
        sample["idx"] = torch.arange(idx_lagged, idx + 1)

        for input_idx in range(1, self.df_meta_data.shape[1]):
            if list(self.df_meta_data.columns)[input_idx] == "measurements":
                continue
            shape = [self.seq_len] + self.data_shapes[input_idx - 1]
            data = np.zeros(shape)
            idx_array = 0
            for data_point_idx in range(idx_lagged, idx + 1):
                file_path = self.__get_file_path_from_df(input_idx, data_point_idx)
                data_t = self.load_data_from_path(file_path)
                data[idx_array] = data_t
                idx_array += 1
            sensor = self.df_meta_data.columns[input_idx]
            sample[sensor] = data
        if "measurements" in self.used_inputs:
            shape = [len(self.used_measurements), self.seq_len]
            data = np.zeros(shape)
            input_idx = list(self.df_meta_data.columns).index("measurements")
            idx_array = 0
            for data_point_idx in range(idx_lagged, idx + 1):
                file_path = self.__get_file_path_from_df(input_idx, data_point_idx)
                data_t = self.load_data_from_path(file_path)
                for meas_idx, meas in enumerate(self.used_measurements):
                    data[meas_idx, idx_array] = data_t[meas]
                idx_array += 1
            for meas_idx, meas in enumerate(self.used_measurements):
                sample[meas] = data[meas_idx]

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
            num_entries = np.array([len(sublist) for sublist in meta_data_route])
            if not np.all(num_entries == num_entries[0]):
                print(f"{route_folder} has varying number of data files among input folders. It got discarded from the dataset.")
                continue
            df_meta = pd.concat([df_meta, pd.DataFrame(columns=columns, data=np.transpose(meta_data_route))], ignore_index=True)        
        return df_meta
    
    def __get_file_path_from_df(self, input_idx, data_point_idx):
        route = self.df_meta_data.iloc[data_point_idx, 0]
        sensor, file_name = self.df_meta_data.columns[input_idx], self.df_meta_data.iloc[data_point_idx, input_idx]
        file_path = os.path.join(self.root_dir, route, sensor, file_name)
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
        file_format = os.path.splitext(path)[1]
        data = None
        if file_format in [".png", ".jpg"]:
            data = cv2.imread(path)
            # reshape to #channels; height; width
            data = data.reshape([3] + list(data.shape)[:-1])
        elif file_format == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
        elif file_format == ".npy":
            data = np.load(path, allow_pickle=True)
            # discard the weird single number for lidar
            data = data[1]

        return data
    
    def __get_data_shapes(self):
        path_parts = self.df_meta_data.iloc[0].to_numpy()
        input_types = self.df_meta_data.columns.to_numpy()
        shapes = []
        for i in range(1, len(path_parts)):
            path = os.path.join(self.root_dir, path_parts[0], input_types[i], path_parts[i])
            data = self.load_data_from_path(path)
            if isinstance(data, np.ndarray):
                shapes.append(list(data.shape))
            else:
                shapes.append(None)
        return shapes



class CARLADatasetMultiProcessing(Dataset):

    def __init__(self, root_dir, config):
        """
        Args:
            root_dir (string): Directory with all the images.
            used_inputs (list): Contains the folder names of the inputs that shall be used.
            used_measurements (list): Contains the attributes of the measurements that shall be used.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.used_inputs = np.array(config["used_inputs"])# .astype(np.string_)
        self.used_measurements = np.array(config["used_measurements"])# .astype(np.string_)
        self.seq_len = config["seq_len"]
        self.df_meta_data = self.__create_metadata_df(root_dir, self.used_inputs)
        self.data_shapes = self.__get_data_shapes()

    def __len__(self):
        return len(self.df_meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_lagged = idx - (self.seq_len - 1)
        sample = dict()
        sample["idx"] = torch.arange(idx_lagged, idx + 1)

        for input_idx in range(1, self.df_meta_data.shape[1]):
            if list(self.df_meta_data.columns)[input_idx] == "measurements":
                continue
            shape = [dim for dim in self.data_shapes[input_idx - 1] if dim != 0]
            data = np.zeros(shape)
            idx_array = 0
            for data_point_idx in range(idx_lagged, idx + 1):
                file_path = self.__get_file_path_from_df(input_idx, data_point_idx)
                data_t = self.load_data_from_path(file_path)
                data[idx_array] = data_t
                idx_array += 1
            sensor = self.df_meta_data.columns[input_idx]
            sample[sensor] = data
        if "measurements" in self.used_inputs:
            input_idx = list(self.df_meta_data.columns).index("measurements")
            shape = [dim for dim in self.data_shapes[input_idx - 1] if dim != 0]
            data = np.zeros(shape)
            idx_array = 0
            for data_point_idx in range(idx_lagged, idx + 1):
                file_path = self.__get_file_path_from_df(input_idx, data_point_idx)
                data_t = self.load_data_from_path(file_path)
                for meas_idx, meas in enumerate(self.used_measurements):
                    data[meas_idx, idx_array] = data_t[meas]
                idx_array += 1
            for meas_idx, meas in enumerate(self.used_measurements):
                sample[meas] = data[meas_idx]

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
            num_entries = np.array([len(sublist) for sublist in meta_data_route])
            if not np.all(num_entries == num_entries[0]):
                print(f"{route_folder} has varying number of data files among input folders. It got discarded from the dataset.")
                continue
            df_meta = pd.concat([df_meta, pd.DataFrame(columns=columns, data=np.transpose(meta_data_route))], ignore_index=True)        
        return df_meta
    

    def __get_file_path_from_df(self, input_idx, data_point_idx):
        route = self.df_meta_data.iloc[data_point_idx, 0]
        sensor, file_name = self.df_meta_data.columns[input_idx], self.df_meta_data.iloc[data_point_idx, input_idx]
        file_path = os.path.join(self.root_dir, route, sensor, file_name)
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
        file_format = os.path.splitext(path)[1]
        data = None
        if file_format in [".png", ".jpg"]:
            data = cv2.imread(path)
            # reshape to #channels; height; width
            data = data.reshape([3] + list(data.shape)[:-1])
        elif file_format == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
        elif file_format == ".npy":
            data = np.load(path, allow_pickle=True)
            # discard the weird single number for lidar
            data = data[1]

        return data

    def __get_data_shapes(self):
        path_parts = self.df_meta_data.iloc[0].to_numpy()
        input_types = self.df_meta_data.columns.to_numpy()
        shapes = []
        for i in range(1, len(path_parts)):
            path = os.path.join(self.root_dir, path_parts[0], input_types[i], path_parts[i])
            data = self.load_data_from_path(path)
            if isinstance(data, np.ndarray):
                shapes.append([self.seq_len] + list(data.shape))
            # This is then the "measurement" dict
            else:
                shapes.append([len(self.used_measurements), self.seq_len])
        max_dim = max([len(shape) for shape in shapes])
        for idx, shape in enumerate(shapes):
            shapes[idx] = [0]*(max_dim - len(shape)) + shape
        return np.array(shapes).astype(np.int_)
    


# test commit

# path_ege_data = os.path.join("data", "Dataset Ege")

# config = {"used_inputs": ["rgb", "depth", "measurements"], 
#         "used_measurements": ["speed", "steer", "throttle"],
#         "seq_len": 1
#         }

# def transform(sample):
#         """
#         This function reshapes every input in a sample to be
#         shape = [seq_len, data_shape]
#         """
#         for key in sample:
#                 data_tensor = torch.Tensor(sample[key])
#                 shape = list(data_tensor.shape)
#                 shape.remove(config["seq_len"])
#                 shape_new = [config["seq_len"]] + shape
#                 sample[key] = data_tensor.reshape(shape_new)
#         return sample

# # data_transform = transforms.Compose([
# #         torch.stack()
# #         # transforms.ToTensor(),
# #         # torch.reshape(config["seq_len"], ...)
# #         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #         #                      std=[0.229, 0.224, 0.225])
# # ])

# dataset = CARLADataset(root_dir=path_ege_data, config=config, transform=None)
# #weighted_sampler = WeightedSampler(dataset=dataset)
# print(dataset.__len__())




