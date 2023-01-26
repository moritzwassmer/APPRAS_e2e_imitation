import pandas as pd
import os
import json 
from tqdm import tqdm
import cv2


def create_metadata_df_xy(root_dir, x, y):
        """
        Creates the metadata (i.e. filenames) based on the the data root directory.
        This root directory is supposed to contain folders for individual routes and those folder
        contain folders with the respective sensor types (i.e. lidar) which contain the actual data files.
        This function assumes that for all routes the same measurements/sensors types were recorded!
        """
        df_temp_list = []
        df_temp = pd.DataFrame()
        for (root, dirs, files) in os.walk(root_dir, topdown=True):
            # Current folder contains the files
            if not dirs:
                input_type = root.split(os.sep)[-1]
                # New route/szenario
                if df_temp.columns.__contains__(input_type):
                    df_temp_list.append(df_temp)
                    df_temp = pd.DataFrame()
                    df_temp["dir"] = [os.path.join(*root.split(os.sep)[:-1])] * len(files)
                    df_temp[input_type] = sorted(files)
                # Append input type to existing route/szenario
                else:
                    if df_temp.empty:
                        df_temp["dir"] = [os.path.join(*root.split(os.sep)[:-1])] * len(files)
                    if len(files) == len(df_temp):
                        df_temp[input_type] = sorted(files)
                    else:
                        print(f"Varying number files among input types: {root}")
        df_temp_list.append(df_temp)
        df = pd.concat(df_temp_list, axis=0, ignore_index=True)        
        return df[["dir"] + used_inputs]

def create_metadata_df(root_dir, used_inputs):
        """
        Creates the metadata (i.e. filenames) based on the the data root directory.
        This root directory is supposed to contain folders for individual routes and those folder
        contain folders with the respective sensor types (i.e. lidar) which contain the actual data files.
        This function assumes that for all routes the same measurements/sensors types were recorded!
        """
        df_temp_list = []
        df_temp = pd.DataFrame()
        for (root, dirs, files) in os.walk(root_dir, topdown=True):
            # Current folder contains the files
            if not dirs:
                files = [file for file in files if not file.startswith(".")]
                input_type = root.split(os.sep)[-1]
                # New route/szenario
                if df_temp.columns.__contains__(input_type):
                    df_temp_list.append(df_temp)
                    df_temp = pd.DataFrame()
                    df_temp["dir"] = [os.path.join(*root.split(os.sep)[:-1])] * len(files)
                    df_temp[input_type] = sorted(files)
                # Append input type to existing route/szenario
                else:
                    if df_temp.empty:
                        df_temp["dir"] = [os.path.join(*root.split(os.sep)[:-1])] * len(files)
                    if len(files) == len(df_temp):
                        df_temp[input_type] = sorted(files)
                    else:
                        print(f"Varying number files among input types: {root}")
        df_temp_list.append(df_temp)
        df = pd.concat(df_temp_list, axis=0, ignore_index=True)        
        return df[["dir"] + used_inputs]


def train_test_split(df_meta_data, towns=None, train_size_random=0.8, seed=None):
    if towns:
        df_train = df_meta_data[df_meta_data["dir"].str.contains("|".join(towns["train"]))]
        df_test = df_meta_data[df_meta_data["dir"].str.contains("|".join(towns["test"]))]
    return df_train, df_test


def measurements_to_df(dataset):
    idxs, paths, speed, steer, throttle, brake, command = [], [], [], [], [], [], []
    df_meta_data = dataset.df_meta_data
    for idx in tqdm(df_meta_data.index.values):
        path = os.path.join(df_meta_data["dir"][idx], "measurements", df_meta_data["measurements"][idx])
        with open(path, 'r') as f:
            measurements_dict = json.load(f)
        idxs.append(idx)
        paths.append(path)
        speed.append(measurements_dict["speed"])
        command.append(measurements_dict["command"])
        steer.append(measurements_dict["steer"])
        throttle.append(measurements_dict["throttle"])
        brake.append(measurements_dict["brake"])

    df_measurements = pd.DataFrame({"dir": paths, "speed": speed, "command": command, "steer": steer, "throttle": throttle, "brake": brake}, index=list(range(len(speed))))
    # df_measurements.to_pickle("measurements_.pickle")
    return df_measurements


def render_example_video_from_folder_name(df_meta_data, folder="int_u_dataset_23_11", path_out="int_u_dataset_23_11.mp4"):
    route_rand = df_meta_data[df_meta_data["dir"].str.contains(folder)].sample(1)["dir"].iloc[0]
    df_meta_data_filt = df_meta_data[df_meta_data["dir"] == route_rand]

    frame_size = (960, 160)
    # fourcc = cv2.VideoWriter_fourcc(*'AVC1')
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    out = cv2.VideoWriter(path_out, fourcc, 2, frame_size)

    for idx in df_meta_data_filt.index.values:
        path_load = os.path.join(df_meta_data_filt["dir"][idx], "rgb", df_meta_data_filt["rgb"][idx])
        img = cv2.imread(path_load)
        out.write(img)

    out.release()
    cv2.destroyAllWindows()