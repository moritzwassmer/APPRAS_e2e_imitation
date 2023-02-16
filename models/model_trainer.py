import torch
import numpy as np
import pandas as pd
import os
import time
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


"""
Current Limitation:
- can only handle seq_len = 1
- cannot handle waypoints as output so far (not even the DataLoader is capable of that currently)
- weights the individual loss terms equally with the same loss function for all predicted values

Apart from that it can be used for arbitrary defined models as long as...
- models forward function takes it's parameters in the order as the DataLoader outputs them (see x_batch)
- models return values of forward function are sorted as the DataLoader outputs them (see y_batch)


TODO:
- Write further useful information as comment to tensorboard (Train/Test Towns and their data sizes)
- exchange python lists by numpy array for loss storage
- write another version in which loss values are only written asynchronously (check speed-up)?
"""

class ModelTrainer:

    def __init__(self, model, optimizer, loss_fns, loss_fn_weights, n_epochs, dataloader_train, dataloader_test, sample_weights, preprocessing, upload_tensorboard):
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
        self.upload_tensorboard = upload_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
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
        self.dir_experiment_save = os.path.join("experiment_files", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(os.path.join(self.dir_experiment_save, "model_state_dict"))
        os.makedirs(os.path.join(self.dir_experiment_save, "optimizer_state_dict"))
        os.makedirs(os.path.join(self.dir_experiment_save, "stats"))
        print(f"Model will be trained on: {self.device}")


    def run(self):
        """
        Sort the model forward parameters as sorted in the DataLoader for the respective data (x_batch).
        Return the model predictions in the order of the DataLoader for the respective data (y_batch).
        - y quantities are sorted alphabetically in the y_batch.
        Losses are then also sorted as in DatLoader (alphabetically)
        """
        times_epoch, times_forward, times_backward, times_val = [], [], [], []
        print_every = 200
        num_batches_train = len(self.dataloader_train)

        val_loss_min = np.Inf

        train_loss_list = [[] for _ in range(len(self.dataloader_test.dataset.y))]
        val_loss_list = [[] for _ in range(len(self.dataloader_test.dataset.y))]

        writer = SummaryWriter()
        for epoch in range(1, self.n_epochs+1): 
            start_epoch = time.time()           
            running_loss_list = [0] * len(self.dataloader_test.dataset.y)

            print(f'Epoch {epoch}\n')
            
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
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
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

            # Save network if lower validation loss is achieved
            val_loss = np.mean(val_loss_list, axis=0)[-1]
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                path_save_model = os.path.join(self.dir_experiment_save, "model_state_dict", f"{self.model.__class__.__name__}_ep{epoch}.pt".lower())
                path_save_opt = os.path.join(self.dir_experiment_save, "optimizer_state_dict", f"opt_{self.model.__class__.__name__}.pt".lower())
                self.model.cpu()
                torch.save(self.model.state_dict(), path_save_model)
                self.model.to(self.device)
                torch.save(self.optimizer.state_dict(), path_save_opt)

            # Save stats    
            self.df_performance_stats = self.get_performance_stats(train_loss_list, val_loss_list)
            self.df_performance_stats.to_csv(os.path.join(self.dir_experiment_save, "stats", "stats_performance.csv"))
            self.df_speed_stats = self.get_speed_stats(times_epoch, times_forward, times_backward, times_val)
            self.df_speed_stats.to_csv(os.path.join(self.dir_experiment_save, "stats", "stats_speed.csv"))
            self.write_to_tensorboard(writer, train_loss_list, val_loss_list, epoch)
            # Back to training
            self.model.train()
            times_epoch.append(time.time() - start_epoch)
            print("Epoch took: ", str(datetime.timedelta(seconds=int(times_epoch[-1]))))
        writer.close()

        if self.upload_tensorboard:
            self.upload_tensorboard_to_cloud(times_epoch)

    def preprocess_on_the_fly(self, X, Y_true):
        """
        Takes X and Y as dictionaries and returns them as lists (sorted like the dict)
        """
        # Attention: this is the point where the dicts are transferred to lists, 
        # where the elements of the lists are sorted in the previous key orders.
        X["rgb"] = torch.squeeze(X["rgb"])
        X = [self.preprocessing[key](X[key]).float() for key in X]
        Y_true = [Y_true[key].float() for key in Y_true]
        return X, Y_true


    def compute_loss(self, Y_true, Y_pred, IDX, do_weight_samples=True):
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


    def get_dataset_predictions(self):
        y_true_list = [[] for _ in range(len(self.dataloader_test.dataset.y))]
        y_pred_list = [[] for _ in range(len(self.dataloader_test.dataset.y))]
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (X_true, Y_true) in tqdm(enumerate(self.dataloader_test)):
                # Preprocess
                X_true, Y_true = self.preprocess_on_the_fly(X_true, Y_true)
                # Move to device (possibly CUDA)
                X_true = [X_.to(self.device) for X_ in X_true]
                Y_true = [Y_.to(self.device) for Y_ in Y_true]
                # Predict
                Y_pred = self.model(*X_true)
                # Update lists
                Y_pred = [torch_pred.flatten().tolist() for torch_pred in Y_pred]
                Y_true = [torch_true.flatten().tolist() for torch_true in Y_true]
                y_true_list = [old + new for old, new in zip(y_true_list, Y_true)]
                y_pred_list = [old + new for old, new in zip(y_pred_list, Y_pred)]
        return y_true_list, y_pred_list
        

    def write_to_tensorboard(self, writer, train_loss_list, val_loss_list, epoch):
        for idx, output in enumerate(self.dataloader_train.dataset.y):
            writer.add_scalars(f"{self.model.__class__.__name__}: loss_{output}", {
                                    'train': train_loss_list[idx][-1],
                                    'val': val_loss_list[idx][-1],
                                }, epoch)


    def upload_tensorboard_to_cloud(self, times_epoch):
        dirs = os.listdir("runs")
        dirs_creation_time = [os.path.getctime(os.path.join("runs", dir)) for dir in dirs]
        dirs_creation_time_sorted = [el[0] for el in sorted(zip(dirs, dirs_creation_time), key=lambda x: x[1])]
        dir_newest = dirs_creation_time_sorted[-1]
        df_stats_train = self.dataloader_train.dataset.get_statistics()
        df_stats_test = self.dataloader_test.dataset.get_statistics()
        description = f""" 
        Trained on towns: {", ".join(sorted(self.dataloader_train.dataset.df_meta_data["dir"].str.extract(r'(Town[0-9][0-9])')[0].unique()))}
        Trained on size GB: {round(df_stats_train[df_stats_train.columns[:-2]].sum().sum(), 2)}
        Trained on % of entire data: {df_stats_train["%_of_entire_data"].item()}
        Validated on towns: {", ".join(sorted(self.dataloader_test.dataset.df_meta_data["dir"].str.extract(r'(Town[0-9][0-9])')[0].unique()))}
        Validated on size GB: {df_stats_test[df_stats_test.columns[:-2]].sum().sum()}
        Validated on % of entire data: {df_stats_test["%_of_entire_data"].item()}
        Wall-time elapsed: {str(datetime.timedelta(seconds=int(sum(times_epoch))))}
        """

        code = f"""tensorboard dev upload --logdir {os.path.join("runs", dir_newest)} \
        --name "{self.model.__class__.__name__}    Epochs={self.n_epochs}" \
        --description "{description}" \
        --one_shot"""
        os.system(code)


    def get_performance_stats(self, train_loss_list, val_loss_list):
        data = np.vstack((train_loss_list, val_loss_list)).T
        columns = ["train_" + e + "_loss" for e in self.dataloader_test.dataset.y] + \
                    ["val_" + e + "_loss" for e in self.dataloader_test.dataset.y]
        df_performance_stats = pd.DataFrame(data=data, columns=columns)
        return df_performance_stats


    def get_speed_stats(self, times_epoch, times_forward, times_backward, times_val):
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
            