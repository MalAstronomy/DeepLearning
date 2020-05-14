import os
import time
import utils
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from dataset import merger_dataset, splitDataLoader, ToTensor, Normalize
from criterion import *

import numpy as np
import sys

def train(model, dataset, optimizer, criterion, model_name, split=[0.9, 0.1], batch_size=16, 
          n_epochs=1, model_path='./', random_seed=None):
    
    # Create directory if doesn't exist
    model_dir = model_path+model_name 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Logging: we save output mmessages in a log file
    log_path = os.path.join(model_dir, 'logs.log')
    utils.set_logger(log_path)
   
    # Dataset
    dataloaders = {}
    dataloaders['train'], dataloaders['val'] = splitDataLoader(dataset, split=split, 
                                                               batch_size=batch_size, random_seed=random_seed)

    # ---
    # If the validation loss reaches a plateau, we decrease the learning rate:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    #scheduler = CosineWithRestarts(optimizer, T_max=40, eta_min=1e-7, last_epoch=-1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7, last_epoch=-1)
    
    # Metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    
    metrics = {
        'model': model_dir,
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'dataset_size': int(len(dataset)),
        'train_size': int(split[0]*len(dataset)),
        'test_size': int(split[1]*len(dataset)),
        'n_epoch': n_epochs,
        'batch_size': batch_size,
        'learning_rate': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # Device: We use cuda on GPUs only
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    # Training
    since = time.time()
    dataset_size = {
        'train':int(split[0]*len(dataset)),
        'val':int(split[1]*len(dataset))
    }
      
    best_loss = 0.0
    for epoch in range(n_epochs):
        
        logging.info('-'*30)
        epoch_time = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            zernike_loss = 0.0
            
            for _, sample in enumerate(dataloaders[phase]):
                # GPU support 
                inputs = sample['input'].to(device)
                target = sample['target'].to(device)

                ##############################################################
                
                # Zero the parameter gradients
                # The backward() function accumulates gradients -> zero_grad() not to mix up gradients between minibatches
                optimizer.zero_grad()
                
                # forward: track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # 1. Make prediction:
                    ratio_estimation = model(inputs)
                    #print(abs(ratio_estimation - target))
                    # 2. Compute the loss for the current batch:
                    loss = criterion(torch.squeeze(ratio_estimation), torch.squeeze(target))

                    # Perform backward propagation to update the weights:
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   
                    
                    running_loss += 1 * loss.item() * inputs.size(0)
                    
            logging.info('[%i/%i] %s loss: %f' % (epoch+1, n_epochs, phase, running_loss / dataset_size[phase]))
            
            # Update metrics
            metrics[phase+'_loss'].append(running_loss / dataset_size[phase])
            #metrics['zernike_'+phase+'_loss'].append(zernike_loss / dataset_size[phase])
            if phase=='train':
                metrics['learning_rate'].append(get_lr(optimizer))
                
            # Adaptive learning rate
            if phase == 'val':
                #scheduler.step()
                # If scheduler is ReduceLROnPlateau we need to give current validation loss:
                scheduler.step(metrics[phase+'_loss'][epoch])
                # Save weigths
                if epoch == 0 or running_loss < best_loss:
                    best_loss = running_loss
                    model_path = os.path.join(model_dir, 'model.pth')
                    torch.save(model.state_dict(), model_path)
                # Save metrics
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4) 
                    
        logging.info('[%i/%i] Time: %f s' % (epoch + 1, n_epochs, time.time()-epoch_time))
        
    time_elapsed = time.time() - since    
    logging.info('[-----] All epochs completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            
        
def get_lr(optimizer):
    for p in optimizer.param_groups:
        lr = p['lr']
    return lr  