import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__()
        
        self.layer_in = self.layer(n_channels_in, 828)
        self.layer_h1 = self.layer(828, 828)
        self.layer_h2 = self.layer(828, 828)
        self.layer_h3 = self.layer(828, 828)
        self.layer_out = self.layer(828, n_channels_out, DO_prob=0) #no Dropout applied

        
    def layer(self, nb_neurons_in, nb_neurons_out, DO_prob=0.5):
        layer = nn.Sequential(
            nn.Linear(nb_neurons_in, nb_neurons_out, bias=True),
            nn.ReLU(),
            #nn.SELU(),
            nn.Dropout(p=DO_prob))
        return layer
        

    def forward(self, xb):
        x = xb.view(xb.size(0), -1) #Flatten data

        #print(x.size())
        x = self.layer_in(x)
        #print(x.size())
        x = self.layer_h1(x)
        x = self.layer_h2(x)
        #x = self.layer_h3(x)
        r = self.layer_out(x)

        return r