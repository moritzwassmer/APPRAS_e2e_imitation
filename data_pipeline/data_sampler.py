from torch.utils.data import Sampler
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
        """Initializes the instance.

        Args:
            dataset : CARLADataset
            df_measurements : pd.DataFrame
            batch_size : int
        
        Raises:
            AssertionError: If batch-size isn't a multiple of the number of unique commands.
        """

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
