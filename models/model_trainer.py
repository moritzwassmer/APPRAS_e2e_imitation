import torch
import numpy as np
import pandas as pd
import os
import time
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


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

    def __init__(self, model, optimizer, loss_fn, n_epochs, dataloader_train, dataloader_test, preprocessing, upload_tensorboard):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.preprocessing = preprocessing
        self.upload_tensorboard = upload_tensorboard

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
        self.model.to(self.device)
        self.df_performance_stats = None
        self.df_speed_stats = None

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
        total_step = len(self.dataloader_train)

        val_loss_min = np.Inf

        train_loss_list = [[] for _ in range(len(self.dataloader_test.dataset.y))]
        val_loss_list = [[] for _ in range(len(self.dataloader_test.dataset.y))]

        writer = SummaryWriter()
        for epoch in range(1, self.n_epochs+1): 
            start_epoch = time.time()           
            running_loss_list = [0] * len(self.dataloader_test.dataset.y)

            print(f'Epoch {epoch}\n')
            
            # Work through batches
            for batch_idx, (X_true, y_true) in enumerate(self.dataloader_train):
                # forward
                start_forward = time.time()
                loss_list = self.forward_pass(X_true, y_true, writer, epoch)
                loss = sum(loss_list) / 3
                times_forward.append(time.time() - start_forward)
                
                # backpropagation
                start_backward = time.time()
                loss.backward()
                self.optimizer.step()
                running_loss_list = [running_loss + loss_.item() for running_loss, loss_ in zip(running_loss_list, loss_list)]
                

                if (batch_idx) % print_every == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch, self.n_epochs, batch_idx, total_step, loss.item()))
                times_backward.append(time.time() - start_backward)
                
            # Epoch finished, evaluate network and save if network_learned
            [train_loss_list[i].append(running_loss_list[i] / total_step) for i in range(len(running_loss_list))]

            print(f'\ntrain-loss: {np.mean(train_loss_list):.4f},')
            # batch_loss, batch_loss_brake, batch_loss_steer, batch_loss_throttle = 0, 0, 0, 0
            batch_loss_list = [0] * len(self.dataloader_test.dataset.y)
            start_val = time.time()
            with torch.no_grad():
                self.model.eval()
                
                for batch_idx, (X_true, y_true) in enumerate(self.dataloader_test):
                    loss_list = self.forward_pass(X_true, y_true, writer, epoch)
                    batch_loss_list = [batch_loss + loss_.item() for batch_loss, loss_ in zip(batch_loss_list, loss_list)]

                [val_loss_list[i].append(batch_loss_list[i] / len(self.dataloader_test)) for i in range(len(batch_loss_list))]   
                print(f'Validation Loss: {np.mean(val_loss_list):.4f}, \n')
                times_val.append(time.time() - start_val)

            # Save network if lower validation loss is achieved
            val_loss = np.mean(val_loss_list, axis=0)[-1]
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                torch.save(self.model.state_dict(), 'resnet.pt')

            self.write_to_tensorboard(writer, train_loss_list, val_loss_list, epoch)
            # Back to training
            self.model.train()
            times_epoch.append(time.time() - start_epoch)
        writer.close()
        self.df_performance_stats = self.get_performance_stats(train_loss_list, val_loss_list)
        self.df_speed_stats = self.get_speed_stats(times_epoch, times_forward, times_backward, times_val)
        if self.upload_tensorboard:
            self.upload_tensorboard_to_cloud(times_epoch)

    def to_cuda_if_possible(self, data):
        return data.to(self.device) if self.device else data
 
    def forward_pass(self, X_true, Y_true, writer, epoch):
        # (optional) do preprocessing (squeezing is currently done in the models forward function)
        X_true["rgb"] = torch.squeeze(X_true["rgb"])
        X_true = [self.preprocessing[key](X_true[key]).float() for key in X_true]
        Y_true = [Y_true[key].float() for key in Y_true]
        
        # move to cuda
        X_true = [self.to_cuda_if_possible(X_) for X_ in X_true]
        Y_true = [self.to_cuda_if_possible(Y_) for Y_ in Y_true]

        # write model graph
        # if epoch == 1:
        #     writer.add_graph(self.model, *X_true) # throws error

        # forward pass
        self.optimizer.zero_grad()
        Y_pred = self.model(*X_true)
        Y_pred = [self.to_cuda_if_possible(Y_) for Y_ in Y_pred]

        # compute loss
        loss_list = [self.loss_fn(Y_pred[i], Y_true[i]) for i in range(len(Y_true))]
        return loss_list

    def write_to_tensorboard(self, writer, train_loss_list, val_loss_list, epoch):
        for idx, output in enumerate(self.dataloader_train.dataset.y):
            writer.add_scalars(f"{self.model.__class__.__name__}: loss_{output}", {
                                    'train': train_loss_list[idx][-1],
                                    'val': val_loss_list[idx][-1],
                                }, epoch)


    def upload_tensorboard_to_cloud(self, times_epoch):
        dir_newest = sorted(os.listdir("runs"))[0]
        df_stats_train = self.dataloader_train.dataset.get_statistics()
        df_stats_test = self.dataloader_test.dataset.get_statistics()
        description = f""" 
        Trained on towns: {", ".join(sorted(self.dataloader_train.dataset.df_meta_data["dir"].str.extract(r'(Town[0-9][0-9])')[0].unique()))}
        Trained on size GB: {df_stats_train[df_stats_train.columns[:-2]].sum().sum()}
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
            