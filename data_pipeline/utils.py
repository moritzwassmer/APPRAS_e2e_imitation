import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
import os
import json 
from tqdm import tqdm
import cv2


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

def adjust_df_meta_for_unique_command_batches(df_meta_data, batch_size):
    """
    Use this function if an model architecture is used where the prediction
    head depend on the specific command type.
    Returns the given df_meta_data in an adjusted version for the use in combination
    with DataLoader where shuffle=False.
    """
    # Add command as column to df_meta_data
    df_meta_data = df_meta_data.copy()
    print("Read all measurements from disk.")
    df_measurements = measurements_to_df(df_meta_data)
    print("All measurements in RAM! Now rebuild batches!")
    df_meta_data["command"] = df_measurements["command"]
    # Discard samples such that num_sample_per_command modulo batch_size = 0
    nums_to_discard = (df_meta_data.sort_values("command").value_counts("command") % batch_size).sort_index().values
    commands = np.sort(df_meta_data["command"].unique())
    for cmd_idx, cmd in enumerate(commands):
        idxs_discard = (df_meta_data
                        .query("command == @cmd")
                        .sample(nums_to_discard[cmd_idx])
                        .index)
        df_meta_data = df_meta_data.drop(index=idxs_discard)
    # Randomly sample batches of batch_size with same command until all samples are used
    batches_list = []
    num_batches = int(len(df_meta_data) / batch_size)
    # while not df_meta_data.empty:
    for _ in tqdm(range(num_batches)):
        # Next cmd to build batch with
        cmd = df_meta_data.sample()["command"].item()
        df_batch = df_meta_data.query("command == @cmd").sample(batch_size).copy()
        df_meta_data = df_meta_data.drop(index=df_batch.index).reset_index(drop=True)
        batches_list.append(df_batch)
    
    return pd.concat(batches_list, ignore_index=True).drop(columns=["command"])


def train_test_split(df_meta_data, towns_intersect=None, towns_no_intersect=None, df_meta_data_noisy=None):
    
    if towns_intersect:
        train_towns = towns_intersect["train"]
        test_towns  = towns_intersect["test"]
        # 250 yields in combination with random_state=3 ~ 80% train and ~20% test 
        num_routes_from_train_towns_for_test = 250
        df_meta_data_routes = df_meta_data["dir"].drop_duplicates().to_frame()
        df_meta_data_routes["town"] = df_meta_data_routes["dir"].str.extract("(Town[0-9][0-9])")
        # Sort by "dir" and index (which is "dir" in that case) alphabetically & reset_index afterwards such that 
        # on all machines the same routes are sampled for the same seed
        df_meta_data_routes = df_meta_data_routes.sort_values(by="dir").reset_index(drop=True)
        df_meta_data_routes["num_frames"] = df_meta_data["dir"].value_counts().sort_index().reset_index(drop=True)
        df_meta_data_routes_train = df_meta_data_routes[df_meta_data_routes["town"].isin(train_towns)]
        df_meta_data_routes_test_2 = df_meta_data_routes[df_meta_data_routes["town"].isin(test_towns)]
        df_meta_data_routes_test_1 = df_meta_data_routes_train.sample(num_routes_from_train_towns_for_test, random_state=3)
        df_meta_data_routes_train = df_meta_data_routes_train.drop(index=df_meta_data_routes_test_1.index)

        # Also sort entries here for easier comparability
        df_train = df_meta_data[df_meta_data["dir"].isin(df_meta_data_routes_train["dir"])].sort_values(["dir", "measurements"]).reset_index(drop=True)
        if type(df_meta_data_noisy) != type(None):
            df_meta_data_noisy = df_meta_data_noisy.sort_values(["dir", "measurements"]).reset_index(drop=True)
            df_train = pd.concat([df_train, df_meta_data_noisy], ignore_index=True)
        df_test_1 = df_meta_data[df_meta_data["dir"].isin(df_meta_data_routes_test_1["dir"])].sort_values(["dir", "measurements"]).reset_index(drop=True)
        df_test_2 = df_meta_data[df_meta_data["dir"].isin(df_meta_data_routes_test_2["dir"])].sort_values(["dir", "measurements"]).reset_index(drop=True)
        return df_train, df_test_1, df_test_2
    
    
    if type(towns_no_intersect) != type(None):
        df_train = df_meta_data[df_meta_data["dir"].str.contains("|".join(towns_no_intersect["train"]))]
        if df_meta_data_noisy:
            df_train = pd.concat([df_train, df_meta_data_noisy])
        df_test = df_meta_data[df_meta_data["dir"].str.contains("|".join(towns_no_intersect["test"]))]
        # Make test set the 15% of the size of the train set
        df_test = df_test.sample(n=int(df_train.shape[0] * 0.15), random_state=3)
        return df_train, df_test


def measurements_to_df(df_meta_data):
    idxs, paths, speed, steer, throttle, brake, command = [], [], [], [], [], [], []
    # df_meta_data = dataset.df_meta_data
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


def get_sample_weights_of_dataset(dataset, num_bins=5, multilabel_option=False):
    """
    """
    num_bins_usually = num_bins
    df_measurements = measurements_to_df(dataset.df_meta_data)
    y_variables = dataset.y
    if "brake" in y_variables:
        df_measurements["brake"] = df_measurements["brake"].replace({True: 1, False: 0})
    # measurements_dict = df_measurements[y_variables].to_dict(orient="list")
    sample_weights = dict()
    for y_var in y_variables:
        num_bins = num_bins_usually
        if y_var == "brake":
            num_bins = 2
        # Create bins in which to classify
        _, bin_edges = np.histogram(df_measurements[y_var], bins=num_bins)
        # Adjust last bin edge such that all data samples are classified
        bin_edges[-1] += 0.1
        # Classify in the defined bins (bin index)
        bin_mapping = np.digitize(df_measurements[y_var], bins=bin_edges,)
        sample_weights_y = compute_sample_weight(y=bin_mapping, class_weight="balanced")
        sample_weights[y_var] = sample_weights_y

    if multilabel_option:
        sample_weights = np.prod(np.vstack(sample_weights.values()).T, axis=1)
        sample_weights = {"multilabel": sample_weights}
    return sample_weights
