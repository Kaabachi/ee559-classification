import torch

import dlc_practical_prologue as prologue
import math
from utils import *
from modules import *

torch.set_grad_enabled(False)

# Loading the data
train_input,train_target = generate_disc_set(1000)
test_input,test_target = generate_disc_set(1000)

train_input,train_target = generate_disc_set(1000)
test_input,test_target = generate_disc_set(1000)

# Normalizing inputs
mu,std = train_input.mean(0), train_input.std(0)
train_input.sub_(mu).div_(std)

mu,std = test_input.mean(0), test_input.std(0)
test_input.sub_(mu).div_(std)

# convert target to one-hot encoding
train_target = target_to_onehot(train_target)
test_target = target_to_onehot(test_target)

# Training the model
batch_size = 100
n_epochs = 250
def train_model(model,train_input,train_target):
    criterion = MSELoss()
    optimizer = SGD(model.param(),lr = 0.01)
    for e in range(0,n_epochs):
        for input, targets in zip(train_input.split(batch_size),train_target.split(batch_size)):
            output = model(input)
            loss = criterion(output,targets)
            optimizer.zero_grad()
            model.backward(criterion.dloss(output,targets))
            optimizer.step()
        if e%10==0 :
            print('epoch: ',e,' loss: ',loss)

# defining the model
model = Sequential((Linear(2,25),ReLU(),Linear(25,25),ReLU(),Linear(25,25),ReLU(),Linear(25,25),ReLU(),Linear(25,2)))
train_model(model,train_input,train_target)

# compute errors
print("train error : ", compute_nb_errors(model,train_input,train_target)/1000 * 100 , "%")
print("test error : ", compute_nb_errors(model,test_input,test_target)/1000 * 100,"%")
