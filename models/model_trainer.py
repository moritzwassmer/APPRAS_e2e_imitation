import torch
import numpy as np
import pandas as pd
import os
import time
import datetime
import torch



class ModelTrainer:
    """
    This class implements the PyTorch training loop and can thus be used to train a model with the 
    additional features of saving models in dedicated experiment folders including performance statistics.
    """


    def __init__(self, model, optimizer, loss_fns, loss_fn_weights, n_epochs, dataloader_train, dataloader_test, sample_weights, preprocessing):
        """
        Args:
            model : pd.DataFrame
                DataFrame that contains all information to build paths.
            optimizer : torch.optim
                PyTorch Optimizer that is used during training.
            loss_fns : dict
                Keys are names of target values to predict (exactly as named when returned by the DataLoader)
                in alphabetical order. Values are the respective loss functions of type torch.nn.modules.loss. 
            loss_fn_weights : dict
                Keys are names of target values to predict (exactly as named when returned by the DataLoader)
                in alphabetical order. Values are the respective lost weights (floats). 
            n_epochs : int
                Number of epochs to train the model.
            dataloader_train : torch.DataLoader
                DataLoader containing the training data.
            dataloader_test : torch.DataLoader
                DataLoader containing the test data.
            sample_weights : dict
                DataLoader containing the test data.
            preprocessing : dict
                Preprocessing dictionary defined in data_pipeline.data_preprocessing.
        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fns = loss_fns
        self.n_epochs = n_epochs
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.preprocessing = preprocessing
        # Sample weights dict contains weights alphabetically to y variables
        self.sample_weights = sample_weights
        self.loss_fn_weights = loss_fn_weights
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
        #self.device = torch.device("cpu")
        self.model.to(self.device)
        # Put sample weights and loss function weights to device 
        self.loss_fns = [self.loss_fns[key] for key in self.loss_fns]
        if self.sample_weights:
            self.sample_weights = [torch.tensor(self.sample_weights[key], device=self.device, dtype=torch.float32) for key in self.sample_weights]
        self.loss_fn_weights = [torch.tensor(self.loss_fn_weights[key], device=self.device, dtype=torch.float32) for key in self.loss_fn_weights]
        self.do_weight_samples = True if sample_weights else False
        self.do_predict_waypoints = True if "waypoints" in dataloader_train.dataset.y else False
        self.df_performance_stats = None
        self.df_speed_stats = None
        if not os.path.exists("experiment_files"):
            os.makedirs("experiment_files")
        self.dir_experiment_save = os.path.join("experiment_files", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(os.path.join(self.dir_experiment_save, "model_state_dict"))
        os.makedirs(os.path.join(self.dir_experiment_save, "optimizer_state_dict"))
        os.makedirs(os.path.join(self.dir_experiment_save, "stats"))
        print(f"Model will be trained on: {self.device}")


    def run(self):
        """Executes the training as configured in the constructor."""
        times_epoch, times_forward, times_backward, times_val = [], [], [], []
        print_every = 200
        num_batches_train = len(self.dataloader_train)


        train_loss_list = [[] for _ in range(len(self.dataloader_test.dataset.y))]
        val_loss_list = [[] for _ in range(len(self.dataloader_test.dataset.y))]

        for epoch in range(1, self.n_epochs+1): 
            start_epoch = time.time()           
            running_loss_list = [0] * len(self.dataloader_test.dataset.y)

            print(f'Epoch {epoch}\n')
            self.TRAIN = True
            if self.TRAIN:
                # Work through batches
                for batch_idx, (X, Y_true, IDX) in enumerate(self.dataloader_train):
                    start_forward = time.time()
                    # In this step dicts are transformed to lists with same order
                    X, Y_true = self.preprocess_on_the_fly(X, Y_true)
                    # Move X, Y_true to device
                    X = [X_.to(self.device) for X_ in X]
                    Y_true = [Y_.to(self.device) for Y_ in Y_true]
                    # Y_pred will be on the device where also model and X are
                    Y_pred = self.model(*X)
                    # Individual losses are already weighted by loss_fn_weights
                    loss_list = self.compute_loss(Y_true, Y_pred, IDX, do_weight_samples=self.do_weight_samples)
                    # Normalizing only necessary if loss_fn_weights don't sum to 1
                    loss = sum(loss_list) / sum(self.loss_fn_weights)
                    # Set gradients to None to not accumulate them over iteration (more efficient than optimizer.zero_grad())
                    for param in self.model.parameters():
                        param.grad = None
                    times_forward.append(time.time() - start_forward)
                    # Backpropagation & Step
                    start_backward = time.time()
                    loss.backward()
                    self.optimizer.step()
                    running_loss_list = [running_loss + loss_.item() for running_loss, loss_ in zip(running_loss_list, loss_list)]
                
                    if (batch_idx) % print_every == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch, self.n_epochs, batch_idx, num_batches_train, loss.item()))
                    times_backward.append(time.time() - start_backward)
                [train_loss_list[i].append(running_loss_list[i] / num_batches_train) for i in range(len(running_loss_list))]

                # TODO: Prints the running loss
                train_loss_list_np = np.array(train_loss_list)
                print(f'\nTrain Loss Individual: {train_loss_list_np[:,-1].round(4)}  Train Loss Total: {(train_loss_list_np[:,-1].sum() / sum(self.loss_fn_weights)):.4f}')
            
            # Validate the network
            batch_loss_list = [0] * len(self.dataloader_test.dataset.y)
            start_val = time.time()
            with torch.no_grad():
                self.model.eval()
                
                for batch_idx, (X, Y_true, IDX) in enumerate(self.dataloader_test):
                    # In this step dicts are transformed to lists with same order
                    X, Y_true = self.preprocess_on_the_fly(X, Y_true)
                    # Move X, Y_true to device
                    X = [X_.to(self.device) for X_ in X]
                    Y_true = [Y_.to(self.device) for Y_ in Y_true]
                    # Y_pred will be on the device where also model and X are
                    Y_pred = self.model(*X)
                    loss_list = self.compute_loss(Y_true, Y_pred, IDX, do_weight_samples=False)
                    loss = sum(loss_list) / sum(self.loss_fn_weights)
                    # TODO: Is it really needed if model.eval() anyways?
                    for param in self.model.parameters():
                        param.grad = None
                    batch_loss_list = [batch_loss + loss_.item() for batch_loss, loss_ in zip(batch_loss_list, loss_list)]

                [val_loss_list[i].append(batch_loss_list[i] / len(self.dataloader_test)) for i in range(len(batch_loss_list))]   
                val_loss_list_np = np.array(val_loss_list)
                print(f'Val Loss Individual: {val_loss_list_np[:,-1].round(4)}  Val Loss Total: {(val_loss_list_np[:,-1].sum() / sum(self.loss_fn_weights)):.4f}\n')
                times_val.append(time.time() - start_val)


            if self.TRAIN:
                path_save_opt = os.path.join(self.dir_experiment_save, "optimizer_state_dict", f"opt_{self.model.__class__.__name__}.pt".lower())
                torch.save(self.optimizer.state_dict(), path_save_opt)
                # TODO: To be moved in if block again
                path_save_model = os.path.join(self.dir_experiment_save, "model_state_dict", f"{self.model.__class__.__name__}_ep{epoch}.pt".lower())
                self.model.cpu()
                torch.save(self.model.state_dict(), path_save_model)
                self.model.to(self.device)
            
            # Save stats    
            self.df_performance_stats = self.get_performance_stats(train_loss_list, val_loss_list)
            self.df_performance_stats.to_csv(os.path.join(self.dir_experiment_save, "stats", "stats_performance.csv"))
            self.df_speed_stats = self.get_speed_stats(times_epoch, times_forward, times_backward, times_val)
            self.df_speed_stats.to_csv(os.path.join(self.dir_experiment_save, "stats", "stats_speed.csv"))
            # Back to training
            self.model.train()
            times_epoch.append(time.time() - start_epoch)
            print("Epoch took: ", str(datetime.timedelta(seconds=int(times_epoch[-1]))))


    def preprocess_on_the_fly(self, X, Y_true):
        """Performs preprocessing for a given batch on the fly.
        Args:
            X : dict
                X batch returned by torch.DataLoader.
            Y_true : dict 
                Y_true batch returned by torch.DataLoader.
        Returns:
            X : list
                List contains all values in same order of X dict, but preprocessed.
            Y : list
                List contains all values in same order of X dict, but converted to floats.
        """
        X["rgb"] = torch.squeeze(X["rgb"])
        if "lidar_bev" in X.keys():
            X["lidar_bev"] = torch.squeeze(X["lidar_bev"])
        # Attention: this is the point where the dicts are transferred to lists, 
        # where the elements of the lists are sorted in the previous key orders.
        X = [self.preprocessing[key](X[key]).float() for key in X]
        Y_true = [Y_true[key].float() for key in Y_true]
        return X, Y_true


    def compute_loss(self, Y_true, Y_pred, IDX, do_weight_samples=False):
        """ Computes the loss for a given batch. The function also applies the weighting for the loss terms
        and optionally sample weighting for tackling class imbalance.
        Args:
            Y_true : list
                Batch of target values.
            Y_pred : dict 
                Batch of predictions.
            IDX : torch.tensor
                Indices of the samples of the batch.
            do_weight_samples : bool (optional)
                If True, sample weights are applied to loss function.
        Returns:
            loss_list : list
                List contains already loss term weighted and (optionally) sample weighted losses.
        """
        if do_weight_samples:
            # Weight the individual loss terms by their sample weights
            loss_list = [(self.sample_weights[i][IDX] * self.loss_fns[i](Y_pred[i], Y_true[i])).sum() / self.sample_weights[i][IDX].sum() for i in range(len(Y_true))]
        # Equally weight, i.e. reduce mean
        else:
            if self.do_predict_waypoints:
                # Y_true will only have length 1
                loss_list = [self.loss_fns[0](Y_pred, Y_true[0]).mean()]
            else:
                loss_list = [self.loss_fns[i](Y_pred[i], Y_true[i]).mean() for i in range(len(Y_true))]
        # Weight each individual loss term by it's loss weight
        loss_list = [self.loss_fn_weights[i] * loss_list[i] for i in range(len(loss_list))]
        return loss_list
    

    def get_performance_stats(self, train_loss_list, val_loss_list):
        """ Builds a DataFrame of the performance statistics i.e. the train & val losses.
        Args:
            train_loss_list : list
                Contains the train losses for each individual loss term.
            train_loss_list : list
                Contains the val losses for each individual loss term.
        Returns:
            df_performance_stats : pd.DataFrame
                DataFrame containing the individual losses as columns.
        """
        if self.TRAIN:
            data = np.vstack((train_loss_list, val_loss_list)).T
        else:
            train_loss_list = np.empty(np.array(val_loss_list).shape)
            train_loss_list.fill(np.nan)
            data = np.vstack((train_loss_list, val_loss_list)).T

        columns = ["train_" + e + "_loss" for e in self.dataloader_test.dataset.y] + \
                    ["val_" + e + "_loss" for e in self.dataloader_test.dataset.y]
        df_performance_stats = pd.DataFrame(data=data, columns=columns)
        return df_performance_stats


    def get_speed_stats(self, times_epoch, times_forward, times_backward, times_val):
        """ Builds a DataFrame of the speed statistics. Disclaimer: The timing results
        must be interpreted with caution as no specific synchronization was applied
        in the trainings loop.
        Args:
            times_epoch : list
                Contains the time needed for every epoch.
            times_forward : list
                Contains the time needed for every forward pass of each epoch.
            times_backward : list
                Contains the time needed for every backward pass of each epoch.
            times_val : list
                Contains the time needed for the entire validation step in each epoch.
        Returns:
            df_speed_stats : pd.DataFrame
                DataFrame containing timing statistics.
        """
        df_speed_stats = pd.DataFrame({ 
        "times_forward": times_forward, 
        "times_backward": times_backward, 
        })
        df_speed_stats = df_speed_stats.sum().to_frame().T
        df_speed_stats["time_val"] = sum(times_val)
        df_speed_stats["time_untracked"] = sum(times_epoch) - df_speed_stats.sum().sum()
        df_speed_stats = df_speed_stats.T
        df_speed_stats.columns = ["time_sec"]
        df_speed_stats["time_%"] = (df_speed_stats["time_sec"] / df_speed_stats["time_sec"].sum() * 100).round(2)
        df_speed_stats = df_speed_stats.sort_values(by="time_%", ascending=False)
        return df_speed_stats
            