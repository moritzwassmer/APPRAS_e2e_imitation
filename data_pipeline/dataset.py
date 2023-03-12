from torch.utils.data import Dataset
import datetime
import torch
import numpy as np
import os
import cv2
import json


class CARLADataset(Dataset):
    """
    This class is used to build the training dataset and the test dataset for model training.
    In contrast to CARLADataset, this class returns the three variables X, Y, and IDX
    instead of only one variable containing the entire batch, when iterated over.
    """

    def __init__(self, df_meta_data, config):
        """
        Args:
            df_meta_data : pd.DataFrame
                DataFrame that contains all information to build paths.
            config : dict
                Example: config_xy = {"used_inputs": ["rgb", "lidar_bev", "measurements"], 
                            "used_measurements": ["speed", "steer", "throttle", "brake", "command"],
                            "y": ["brake", "steer", "throttle"],
                            "seq_len": 1
                            }
                "used_inputs" states the directory names of each route that shall be used.
                "used_measurements" states the keys, i.e. the quantities from the measurements.json 
                    files that shall be loaded.
                "y" states what which quantities from "used_inputs" and "used_measurements" are considered
                    output variables that shall be predicted by the model.
                "seq_len" states the number of data points that should be fetched for one sample. > 1 only,
                    useful for models involving time convolution etc.. Default should be 1! 
        """
        self.used_inputs = config["used_inputs"]
        self.used_measurements = config["used_measurements"]
        self.seq_len = config["seq_len"]
        self.df_meta_data = df_meta_data
        self.data_shapes = self.__get_data_shapes()

    def __len__(self):
        return len(self.df_meta_data)

    def get_statistics(self):
        """
        Returns:
            df_stats : pd.DataFrame
                Data Frame containing stats about the CARLADatasetXY.
        """
        df_meta_data = self.df_meta_data
        df_meta_data_full_paths = df_meta_data[df_meta_data.columns[1:]].apply(lambda x: df_meta_data["dir"] + os.sep + x.name + os.sep + x)
        df_meta_data_sizes = df_meta_data_full_paths.applymap(lambda path: os.path.getsize(path))
        df_stats = (df_meta_data_sizes.sum() / 10**9).round(2).to_frame().T # 10**9 or GB instead of GiB # 1073741824
        df_stats.columns = df_stats.columns + "_in_GB" 
        # df_stats["time_hours"] = len(df_meta_data) / (2 * 60 * 60)
        df_stats["driving_time"] = str(datetime.timedelta(seconds=int(len(df_meta_data) / 2)))
        df_stats["%_of_entire_data"] = round((len(df_meta_data) / 258866 * 100), 2)
        return df_stats

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
                file_path = self.get_file_path_from_df(input_idx, data_point_idx)
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
                file_path = self.get_file_path_from_df(input_idx, data_point_idx)
                data_t = self.load_data_from_path(file_path)
                for meas_idx, meas in enumerate(self.used_measurements):
                    if isinstance(data_t[meas], list):
                        data[meas_idx, idx_array] = data_t[meas][0]
                    else:
                        data[meas_idx, idx_array] = data_t[meas]
                idx_array += 1
            for meas_idx, meas in enumerate(self.used_measurements):
                sample[meas] = data[meas_idx]

        return sample
            
    def get_file_path_from_df(self, input_idx, data_point_idx):
        """ Builds the path that point to a data point.
        Args:
           input_idx : int
                Column index of df_meta_data.
           data_point_idx : int
                Row index of df_meta_data.
        Returns:
            file_path : string
                Path that points to a data point.
        """
        route = self.df_meta_data.iloc[data_point_idx, 0]
        sensor, file_name = self.df_meta_data.columns[input_idx], self.df_meta_data.iloc[data_point_idx, input_idx]
        # file_path = os.path.join(self.root_dir, route, sensor, file_name)
        file_path = os.path.join(route, sensor, file_name)
        return file_path

    def load_data_from_path(self, path):
        """ Loads data from a given path that points to a file in the dataset.
        Args: 
            path : string
                Path to a file.
        Return:
            data : np.ndarray
                Whatever data the given path points to.
        """
        if not os.path.isfile(path):
            print(f"Path is not a file: {path}")
        file_format = os.path.splitext(path)[1]
        data = None
        if file_format in [".png", ".jpg"]:
            data = cv2.imread(path)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            # Reshape to C, H, W
            data = np.transpose(data, (2, 0, 1))

        elif file_format == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
        elif file_format == ".npy":
            data = np.load(path, allow_pickle=True)
            # if lidar
            if data.shape == (2, ):
                data = data[1]
        elif file_format == ".npz":
            with np.load(path, allow_pickle=True) as f:
                data = f["arr_0"]
        return data

    def __get_data_shapes(self):
        """ Return shapes of data that is used according to the config.
        Return:
            shapes : list
                List of shapes.
        """
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
