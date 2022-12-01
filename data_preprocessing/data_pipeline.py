from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import os
import cv2
import json

"""
Inspiration from:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

TODO:
1. Only integrate attributes from the measurements dict that we actually use in the sample dict
    - could be done in the "transform" method that gets applied on the sample
    - could also be applied before because this is rather "selecting" than "transforming"
2. Test CARLADataset with DataLoader (i.e. create batches) (--> no need to have sequential data in the batches)
3. Implement sequential sampling 
4. Test CARLADataset with DataLoader (i.e. create batches) (--> have sequential data in the batches)
"""

class CARLADataset(Dataset):

    def __init__(self, root_dir, used_inputs, used_measurements, transform=None):
        """
        Args:
            used_inputs (list): Contains the folder names of the inputs that shall be used.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df_meta_data = self.__create_metadata_df(root_dir, used_inputs)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # joined_path = os.path.join(self.root_dir, "rgb")
        # len([name for name in os.listdir(joined_path) if os.path.isfile(os.path.join(joined_path, name))])
        return len(self.df_meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        # file_paths = [os.path.join(self.root_dir, self.df_meta_data.columns[j], self.df_meta_data.iloc[idx, j]) \
        #         for j in range(1, self.df_meta_data.shape[1])]
        sample = dict()
        route = self.df_meta_data.iloc[idx, 0]
        for j in range(1, self.df_meta_data.shape[1]):
            sensor, file_name = self.df_meta_data.columns[j], self.df_meta_data.iloc[idx, j]
            file_path = os.path.join(self.root_dir, route, sensor, file_name)
            sample[sensor] = self.load_data_from_path(file_path)

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

    def save_meta_data(self, path=None):
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

        return data

                




# create_metadata_csv("myfolder/sample_trainingsdata")

path_moritz_data = "myfolder/sample_trainingsdata"
path_ege_data = "data/Dataset Ege"


cd = CARLADataset(path_ege_data, ["rgb", "lidar", "depth", "measurements"])
print(cd.__len__())
print(cd.__getitem__(5))



