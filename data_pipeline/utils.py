import pandas as pd
import os 

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