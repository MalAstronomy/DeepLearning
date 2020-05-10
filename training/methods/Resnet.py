import torch
import numpy as np
import aotools
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):

    def __init__(self, n_channels_in, n_channels_out, gridsize):
        super(Net, self).__init__()

        self.resnet = models.resnet50(pretrained=True)   
        
        for param in self.resnet.parameters():
            param.requires_grad = True
       
        # Input size 2x128x128 -> 2x224x224
        
        first_conv_layer = [nn.Conv2d(n_channels_in, 3, kernel_size=1, stride=1, bias=True),
                            nn.AdaptiveMaxPool2d(224),
                            self.resnet.conv1]
        self.resnet.conv1= nn.Sequential(*first_conv_layer)

        # Fit classifier
        self.resnet.fc = nn.Sequential(
                                nn.Linear(2048, n_channels_out)         
                            )    
    
        self.phase2dlayer = Phase2DLayer(n_channels_out, gridsize)

    def forward(self, x):
        # 128x128x2
        z = self.resnet(x)
        return phase, z  
