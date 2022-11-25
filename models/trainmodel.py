import  torch.nn as nn
import torch as torch 
import firstarchitecture as model
import sys

##import training data from relevant dir as x_train, y_train

##make readme of hyperparameters

loss_function = nn.MSELoss()# or MSE makes more sense?

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

epochs=500

final_losses=[]

for i in range(epochs):

    i = i+1

    y_pred=model.forward(X_train)

    loss=loss_function(y_pred,y_train)

    final_losses.append(loss)

    #if i % 10 == 1:

    # print("Epoch number: {} and the loss : {}".format(i,loss.item()))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
