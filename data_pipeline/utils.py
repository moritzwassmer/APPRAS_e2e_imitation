import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
import os
import json 
from tqdm import tqdm
import cv2


def create_metadata_df(root_dir, used_inputs):
    """Creates the metadata (i.e. filenames) based on the the data root directory.
    This root directory is supposed to contain folders for individual routes and those folder
    contain folders with the respective sensor types (i.e. lidar) which contain the actual data files.
    This function assumes that for all routes the same measurements/sensors types were recorded!

    Args:
        root_dir : string
            The path to a data directory with the structure defined by the TransFuser dataset.
        used_inputs : list
            List containing the directories used per scenario e.g. ["rgb", "measurements"]
    
    Returns:
        df : pd.DataFrame
            DataFrame that contains all information to build paths to all files stated in used_inputs.
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



def train_test_split(df_meta_data, towns_intersect=None, towns_no_intersect=None, df_meta_data_noisy=None):
    """ Performs train-test-split on the df_meta_data frame.
    Args:
        df_meta_data : pd.DataFrame
            DataFrame that contains all information to build paths.
        towns_intersect : dict, optional
            Dictionary of the form e.g. {"train" : ["Town04", "Town05"], "test" : ["Town06"]}. (default=None)
            If not None, test set 1 will contain some routes of towns stated in "train" and test set 2 will
            contain all towns as usual stated in "test".
        towns_no_intersect : dict, optional
            Dictionary of the form e.g. {"train" : ["Town04", "Town05"], "test" : ["Town06"]}. (default=None)
            Creates training set and test set according to this dict.
        df_meta_data_noisy : pd.DataFrame, optional
            DataFrame that contains all information to build paths for noisy data. (default=None)
     Returns:
        In case towns_intersect is not None,
        df_train : pd.DataFrame
            DataFrame that contains all information to build paths for training set.
        df_test_1 : pd.DataFrame
            DataFrame that contains all information to build paths for test set 1.
        df_test_2 : pd.DataFrame
            DataFrame that contains all information to build paths for test set 2.
        
        In case towns_no_intersect is not None,
        if towns_no_intersect:
        df_train : pd.DataFrame
            DataFrame that contains all information to build paths for training set.
        df_test : pd.DataFrame
            DataFrame that contains all information to build paths for test set.
    """
    
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
        df_test_1 = df_meta_data[df_meta_data["dir"].isin(df_meta_data_routes_test_1["dir"])].sort_values(["dir", "measurements"]).reset_index(drop=True)
        df_test_2 = df_meta_data[df_meta_data["dir"].isin(df_meta_data_routes_test_2["dir"])].sort_values(["dir", "measurements"]).reset_index(drop=True)
        if type(df_meta_data_noisy) != type(None):
            # Extract route attributes from df_meta_data_noisy
            df_meta_data_noisy_routes = df_meta_data_noisy["dir"].drop_duplicates().reset_index(drop=True)
            df_meta_data_noisy_routes_attr = pd.DataFrame(data=df_meta_data_noisy_routes.str.extract("(Town\d\d)").values, columns=["town"])
            df_meta_data_noisy_routes_attr["scenario"] = df_meta_data_noisy_routes.str.extract("(Scenario\d)").values
            df_meta_data_noisy_routes_attr["route"] = df_meta_data_noisy_routes.str.extract("(route.*?)_").values
            df_meta_data_noisy_routes_attr["dir"] = df_meta_data_noisy_routes
            # Extract route attributes from df_test_1
            df_meta_data_test_routes = df_test_1["dir"].drop_duplicates().reset_index(drop=True)
            df_meta_data_test_routes_attr = pd.DataFrame(data=df_meta_data_test_routes.str.extract("(Town\d\d)").values, columns=["town"])
            df_meta_data_test_routes_attr["scenario"] = df_meta_data_test_routes.str.extract("(Scenario.*?)_").values
            df_meta_data_test_routes_attr["route"] = df_meta_data_test_routes.str.extract("(route.*?)_").values
            # Discard the overlap from the noisy data such that test set gets not corrupted
            overlap_dirs = df_meta_data_test_routes_attr.merge(df_meta_data_noisy_routes_attr, on=["town", "route"])["dir"]
            df_meta_data_noisy = df_meta_data_noisy[~df_meta_data_noisy["dir"].isin(overlap_dirs)].copy()
            df_meta_data_noisy = df_meta_data_noisy.sort_values(["dir", "measurements"]).reset_index(drop=True)
            df_train = pd.concat([df_train, df_meta_data_noisy], ignore_index=True)

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
    """ Reads the following quantities from the .json paths saved in df_meta_data and puts them into a DataFrame:
        speed, steer, throttle, brake, command, theta, x, y, x_command, y_command, waypoints. This function may be
        used to compute sample weights.
    Args:
        df_meta_data : pd.DataFrame
            DataFrame that contains all information to build paths.
     Returns:
        df_measurements : pd.DataFrame
            DataFrame that contains all quantities listed in description above.
    """
    idxs, paths, speed, steer, throttle, brake, command, theta, x, y, x_command, y_command, waypoints = [], [], [], [], [], [], [], [], [], [], [], [], []
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
        theta.append(measurements_dict["theta"])
        x.append(measurements_dict["x"])
        y.append(measurements_dict["y"])
        x_command.append(measurements_dict["x_command"])
        y_command.append(measurements_dict["y_command"])
        waypoints.append(measurements_dict["waypoints"])
    df_measurements = pd.DataFrame({"dir": paths, "speed": speed, "command": command, 
                    "steer": steer, "throttle": throttle, "brake": brake,
                    "theta": theta, "x": x, "y": y, "x_command": x_command, "y_command": y_command,
                    "waypoints": waypoints}, index=list(range(len(speed))))
    return df_measurements


def get_sample_weights_of_dataset(dataset, num_bins=10, multilabel_option=False):
    """ Calculates sample weights for a CARLDADatasetXY instance. 
    Args:
        dataset : CARLDADatasetXY
            CARLDADatasetXY instance for which the sample weights should be calculated.
        num_bins : int, optional
            Number of bins is equal to number of equally wide classes in which the continuous
            y variables such as steer should be classified in.
        multilabel_option : bool, optional
            If set to True, individual sample weights of all y variables used in the given 
            dataset instance will be reduced to one weights list. 
     Returns:
        sample_weights : dict
            Contains the sample weights for all y variables used in the given dataset instance.
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
        # Detect ede cases
        boundary_to_edge_cases = np.quantile(sample_weights, 0.999)
        sample_weights_clipped = sample_weights.copy()
        # Clip edge cases weight to the boundary weight, such that theses cases don't dominate too severely
        sample_weights_clipped[np.argwhere(sample_weights_clipped > boundary_to_edge_cases).flatten()] = boundary_to_edge_cases
        sample_weights = {"multilabel": sample_weights_clipped}
    return sample_weights

def render_example_video_from_folder_name(df_meta_data, folder="int_u_dataset_23_11", path_out="int_u_dataset_23_11.mp4"):
    route_rand = df_meta_data[df_meta_data["dir"].str.contains(folder)].sample(1)["dir"].iloc[0]
    df_meta_data_filt = df_meta_data[df_meta_data["dir"] == route_rand]

    frame_size = (960, 160)
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    out = cv2.VideoWriter(path_out, fourcc, 2, frame_size)

    for idx in df_meta_data_filt.index.values:
        path_load = os.path.join(df_meta_data_filt["dir"][idx], "rgb", df_meta_data_filt["rgb"][idx])
        img = cv2.imread(path_load)
        out.write(img)

    out.release()
    cv2.destroyAllWindows()


def render_example_video_from_folder_name_2(path_to_route, path_out):
    file_paths = sorted(os.listdir(path_out))
    
    frame_size = (960, 160)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(path_out, fourcc, 2, frame_size)
    for path in file_paths:
        img = cv2.imread(path)
        out.write(img)

    out.release()
    cv2.destroyAllWindows()