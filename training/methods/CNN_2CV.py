import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_channels_in, n_channels_out):
        super(Net, self).__init__()
        
        ## Convolutional layers ##
        self.layer1 = self.ConvLayer(n_channels_in, 32, strd_conv=2, strd_pool=2)
        self.layer2 = self.ConvLayer(32, 64, strd_conv=2, strd_pool=2)

        ## Fully connected layers ##
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*2**3, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, n_channels_out),
            nn.ReLU())    
        
    def ConvLayer(self, nb_neurons_in, nb_neurons_out, ksize_conv=3, strd_conv=1, pad_conv=0, ksize_pool=3, strd_pool=1, pad_pool=0):
        '''
        Define a convolutional layer
        '''
        layer = nn.Sequential(
            nn.Conv3d(nb_neurons_in, nb_neurons_out, 
                      kernel_size=ksize_conv, stride=strd_conv, padding=pad_conv),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=ksize_pool, stride=strd_pool, padding=pad_pool))
        return layer   

    def forward(self, x):
        '''
        Forward pass
        '''
        out = self.layer1(x)
        out = self.layer2(out)
        print(out.size())
        out = out.view(out.size(0), -1)  # Reshape to a single vector (per element of the batch) for fully convolutional layer
        out = self.fc(out)
        
        return out