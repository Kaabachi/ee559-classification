import torch

import dlc_practical_prologue as prologue
import math
torch.set_grad_enabled(False)

'''
Generates a training and a test set of 1, 000 points sampled uniformly in [0, 1]2 each with a
label 0 if outside the disk centered at (0.5, 0.5) of radius 1/
√2π, and 1 inside.
'''
def generate_disc_set(nb):
    train = torch.empty(nb,2).uniform_(0,1)
    target = (train.sub(0.5).pow(2).sum(1)<torch.empty(nb).fill_((1/(2*math.pi)))).long()
    return train,target

'''
Converts target of shape (1000) to target of shape (1000,2)
one-hot encoding
'''
def target_to_onehot(target):
    res = torch.empty(target.size(0), 2).zero_()
    res.scatter_(1, target.view(-1, 1), 1.0).mul(0.9)
    return res


def compute_nb_errors(model,data_input,data_target,batch_size = 100):
    nb_errors = 0
    for input,targets in zip(data_input.split(batch_size),data_target.split(batch_size)):
        output = model(input)
        _,predicted_classes = torch.max(output,1)
        for i in range(0,output.size(0)):
            if(targets[i][predicted_classes[i]]!=1):
                nb_errors = nb_errors+1
                
    return nb_errors

'''
Plot set
'''
import matplotlib.pyplot as plt
#Plot the dataset
def plot_set(train_input, train_target):
    
    fig, ax = plt.subplots(1, 1)
    
    # plot points with label 1 in black
    ax.scatter(
    train_input[train_target == 1, 0],
    train_input[train_target == 1, 1],
    c = 'black'
    )
    
    #plot points with label 0 in blue
    ax.scatter(
        train_input[train_target == 0, 0],
        train_input[train_target == 0, 1],
        c = 'blue'
    )

    ax.axis([-1, 2, -1, 2])
    

    
    ax.legend(
        ['label 0', 'label 1']
    )
    
    fig.suptitle('Points scattered randomly and their labels')
    fig.savefig('points.png')
    plt.show()
    
