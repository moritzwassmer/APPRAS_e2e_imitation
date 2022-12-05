import torch
from torch.utils.data import Sampler
import numpy as np
import pandas as pd


"""
Inspiration from https://www.kaggle.com/code/shahules/guide-pytorch-data-samplers-sequence-bucketing

Assigns weights/probabilities to which each index in the dataset gets sampled.
Use cases:
1. Set probabilities of indices to zero which don't have enough previous data point in a specific route
accoring to the specified sequence length.
2. Oversampling of rare events/maneuvers such as turns to balance the dataset (for this additional metadata
will be needed).
"""
class WeightedSampler(Sampler):
    
    def __init__(self, dataset):
        
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)
        restricted_idxs = self.__get_restricted_indices()
        weights = np.ones(dataset.__len__()) * 1 / (dataset.__len__() - len(restricted_idxs))
        weights[restricted_idxs] = 0
        
        self.weights = torch.tensor(weights, dtype=torch.double)
        
    def __iter__(self):
        count = 0
        index = [self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            yield index[count]
            count += 1

    def __get_restricted_indices(self):
        """
        Return:
                restricted_idx (list): List contains idxs that shall not be sampled because they don't
                have enough previous/lagged datapoint with respect to seq_len for each route.
        """
        df_meta_data, seq_len = self.dataset.df_meta_data, self.dataset.seq_len
        boarders = df_meta_data["route"].value_counts().sort_index().to_numpy()
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
        