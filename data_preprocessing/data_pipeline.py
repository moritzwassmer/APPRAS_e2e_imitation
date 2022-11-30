from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import os
import io

"""
Inspiration from:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""



class CARLADataset(Dataset):

    def __init__(self, used_inputs, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta_data = self.__create_metadata_csv(root_dir, used_inputs)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # joined_path = os.path.join(self.root_dir, "rgb")
        # len([name for name in os.listdir(joined_path) if os.path.isfile(os.path.join(joined_path, name))])
        return len(self.meta_data)

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
        print(self.meta_data.columns[0])
        file_names = [os.path.join(self.root_dir, self.meta_data.columns[j], self.meta_data.iloc[idx, j]) \
                for j in range(self.meta_data.shape[1])]


        if self.transform:
            sample = self.transform(sample)

        return paths

    def __create_metadata_csv(self, root_dir, used_inputs):
        dirs = os.listdir(root_dir)
        columns = [dir for dir in dirs if not os.path.isfile(os.path.join(root_dir, dir)) and dir in used_inputs]
        data = []
        for col in columns:
            data.append(os.listdir(os.path.join(root_dir, col)))
        df_meta = pd.DataFrame(columns=columns, data=np.transpose(data))
        df_meta.to_csv(os.path.join(root_dir, "meta_data.csv"), index=False)
        return df_meta




create_metadata_csv("myfolder/sample_trainingsdata")

cd = CARLADataset("myfolder/sample_trainingsdata/meta_data.csv", "myfolder/sample_trainingsdata")
print(cd.__len__())
print(cd.__getitem__(5))



