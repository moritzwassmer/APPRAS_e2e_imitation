import torch
from torch.utils.data import Sampler
import numpy as np
from math import floor



class BranchPerCommandSampler(Sampler):
    """
    This class is used to sample data for model architectures that have one output head per command type.
    The data is sampled such that the batch_sizes remain constant (even for the last batch) and the number of samples per command
    within a batch is equal. Thus, oversampling of the the minor command classes and undersampling of the major classes is
    performed. Furthermore, the samples within a batch are sorted by the command type in ascending order.
    The class can be used by setting it's instance as an argument to a pytorch DataLoader object.
    """
    def __init__(self, dataset, df_measurements, batch_size):

        assert_msg = "The batch-size must be chosen as a multiple of the number of unique commands!"
        assert batch_size % df_measurements["command"].nunique() == 0, assert_msg
        
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)
        self.df_meta_data_train = dataset.df_meta_data.copy()
        self.df_meta_data_train["command"] = df_measurements["command"]
        self.df_meta_data_train_grp = self.df_meta_data_train.groupby("command")
        
        
    def __iter__(self):

        num_commands = self.df_meta_data_train["command"].nunique()
        num_samples_per_batch_per_command = int(self.batch_size / num_commands)
        num_samples_adjusted = floor(self.num_samples / self.batch_size) * self.batch_size
        num_batches = int(num_samples_adjusted / self.batch_size)
        num_samples_per_cmd_total = int(num_samples_adjusted / num_commands)

        indices_reorder = []

        for i in range(num_batches):
            for j in range(num_commands):
                idx_start = j*num_samples_per_cmd_total + i*num_samples_per_batch_per_command
                idx_end = idx_start + num_samples_per_batch_per_command
                indices_reorder += list(range(idx_start, idx_end))

        indices = self.df_meta_data_train_grp.sample(num_samples_per_cmd_total, replace=True).sort_values(by="command").iloc[indices_reorder].index
        for idx in indices:
            yield idx

    
    def __len__(self):
        return self.num_samples


class WeightedSampler(Sampler):
    """
    Inspiration from https://www.kaggle.com/code/shahules/guide-pytorch-data-samplers-sequence-bucketing

    Assigns weights/probabilities to which each index in the dataset gets sampled.
    Use cases:
    1. Set probabilities of indices to zero which don't have enough previous data point in a specific route
    accoring to the specified sequence length.
    2. Oversampling of rare events/maneuvers such as turns to balance the dataset (for this additional metadata
    will be needed).
    """
    
    def __init__(self, dataset):
        
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)
        restricted_idxs = self.get_restricted_indices()
        weights = np.ones(dataset.__len__()) * 1 / (dataset.__len__() - len(restricted_idxs))
        weights[restricted_idxs] = 0
        
        self.weights = torch.tensor(weights, dtype=torch.double)
        
    def __iter__(self):
        count = 0
        index = [self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            yield index[count]
            count += 1

    def get_restricted_indices(self):
        """
        Return:
                restricted_idx (list): List contains idxs that shall not be sampled because they don't
                have enough previous/lagged datapoint with respect to seq_len for each route.
        """
        df_meta_data, seq_len = self.dataset.df_meta_data, self.dataset.seq_len
        boarders = df_meta_data["dir"].value_counts().sort_index().to_numpy()
        boarders_cumsum = np.cumsum(boarders) - 1
        boarders_cumsum = np.insert(boarders_cumsum, 0, 0)
        restricted_idxs = []
        for i in range(len(boarders_cumsum) - 1):
            if boarders_cumsum[i+1] - boarders_cumsum[i] < seq_len:
                restricted_idxs += list(range(boarders_cumsum[i+1] + 1))
            else:
                if boarders_cumsum[i] == 0:
                    restricted_idxs += list(range(seq_len))
                else:
                    restricted_idxs += list(range(boarders_cumsum[i] + 1, boarders_cumsum[i] + 1 + seq_len))
        return restricted_idxs

    
    def __len__(self):
        return self.num_samples
        