#!/usr/bin/env python

# Example of network training using Pytorch (1.1) and Cuda (9).

# NB: This code use real time monitoring based on Visdom
# an open source webserver allowing real time monitoring
#
# https://github.com/facebookresearch/visdom
#
# Start the webserver using:
# python -m visdom.server
#
# Access it on: (by default)
# http://localhost:8097

# Global import

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict
import click

# Local import
sys.path.insert(0, './methods/')
# Import the neural network architectures:
from MLP import Net as MLP
#from SCN import Model as SCN
from CNN import Net as CNN

sys.path.insert(0, './src/')
from dataset import *
from train import *
from criterion import MSELoss

#### Input options ######
@click.command()

## Mandatory:
@click.option('-dla','--dl_arch', type=str, 
              help='Machine learning architecture selected', required=True)
@click.option('-dsiz','--dataset_size', type=int, 
              help='Number of image sets used for training and validation', required=True)
@click.option('-floc','--file_location', type=str, 
              help='Location of the training dataset file', required=True)

## Optional:
@click.option('-opt','--optimizer_name', type=str, 
              help='Name of the optimizer', default='Adam')
@click.option('-bs','--batch_size', type=int, 
              help='Batch size', default=64)
@click.option('-lr','--learning_rate', type=str, 
              help='Learning rate', default='1e-3')
@click.option('-nep','--nb_epoch', type=int, 
              help='Number of epochs', default=200)
@click.option('-splt','--split_train', type=str, 
              help='Proportion of images used for training (the rest is for validation)', default='0.9')
              
def training(dataset_size, file_location, dl_arch, optimizer_name, batch_size, learning_rate, nb_epoch, split_train):    

    # Name to give to the model file:
    model_file_name = 'model_'+str(dataset_size)+'merger_'+dl_arch+'_3cv_bs'+str(batch_size)+'_lr'+\
    str(learning_rate)+'_'+str(nb_epoch)+'ep_opt'+str(optimizer_name)+'_split'+split_train+'_noDO'
    
    # Path where the model will be located:
    res_path = '../models/'

    transfo = transforms.Compose([Normalize(), ToTensor()])

    # Create the dataset object:
    dataset = merger_dataset(path_to_file = file_location, 
                              size = int(float(dataset_size)),
                              transform = transfo)

    # Load model architecture:
    if dl_arch == 'mlp':
        model = MLP(70, 2)
    elif dl_arch == 'cnn':
        model = CNN(1, 2)
    elif dl_arch == 'scn':
        model = SCN(1, 2)
    else:
        sys.exit("the model name specified is not valid")

    # Select which criterion to use to compute the loss:
    criterion = MSELoss()

    # Select which optimizer to use:
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=float(learning_rate), momentum=0.9)
    else:
        sys.exit("the optimizer specified is not valid")

    # Move Network to GPU if available:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        model.cuda()
    else:
        model.cpu()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()

    # Launch training script. The network weights are automatically saved 
    # at the end of an epoch (if the test error is reduced). The metrics are also
    # saved at the end of each epoch in JSON format. All outputs are also stored in a 
    # log file.
    #
    # - model = network to train
    # - dataset = dataset object
    # - optimizer = gradient descent optimizer (Adam, SGD, RMSProp)
    # - criterion = loss function
    # - split[x, 1-x] = Division train/test. 'x' is the proportion of the test set.
    # - batch_size = batch size
    # - n_epochs = number of epochs
    # - model_dir = where to save the results
    # - visdom =  enable real time monitoring

    #Launch training:

    train(model, 
          dataset,
          optimizer, 
          criterion,
          model_name = model_file_name,
          split = [float(split_train)/100, 1-float(split_train)/100], # Split between training and validation dataset
          batch_size = batch_size, # number limite by GPU
          n_epochs = nb_epoch,
          model_path = res_path)

if __name__ == '__main__':
    training()

