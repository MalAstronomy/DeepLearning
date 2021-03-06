import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__()
        
        self.layer_in = self.layer(n_channels_in, 300)
        self.layer_h1 = self.layer(300, 300)
        self.layer_h2 = self.layer(300, 300)
        self.layer_out = self.layer(300, n_channels_out, activation=False, dropout=False) #no activation and Dropout applied

        
    def layer(self, nb_neurons_in, nb_neurons_out, activation=True, dropout=True):
        modules = []
        modules.append(nn.Linear(nb_neurons_in, nb_neurons_out, bias=True))
        if activation:
            modules.append(nn.ReLU())
        if dropout:    
            modules.append(nn.Dropout())

        layer = nn.Sequential(*modules)

        return layer
        

    def forward(self, input):
        x = input.view(input.size(0), -1) #Flatten data

        x = self.layer_in(x)
        x = self.layer_h1(x)
        x = self.layer_h2(x)
        x = self.layer_out(x)
        
        x = F.relu(x)
        out = torch.min(x,torch.ones(x.size()))
        
        return out