import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_channels_in, n_channels_out):
        super(Net, self).__init__()
        
        ## Convolutional layers ##
        self.layer1 = self.ConvLayer(n_channels_in, 32, strd_conv=2)
        self.layer2 = self.ConvLayer(32, 64, strd_conv=2)
        self.layer3 = self.ConvLayer(64, 128, strd_conv=2)
        self.layer4 = self.ConvLayer(128, 256, strd_conv=1)

        
        ## Fully connected layers ##
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128*4**3, 128),
            #nn.SELU(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 128),
            #nn.SELU(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 128),
            #nn.SELU(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, n_channels_out),
            #nn.SELU())
            nn.ReLU())    
        
    def ConvLayer(self, nb_neurons_in, nb_neurons_out, ksize_conv=3, strd_conv=1, pad_conv=0, ksize_pool=3, strd_pool=1, pad_pool=0):
        '''
        Define a convolutional layer
        '''
        layer = nn.Sequential(
            nn.Conv3d(nb_neurons_in, nb_neurons_out, 
                      kernel_size=ksize_conv, stride=strd_conv, padding=pad_conv),
            #nn.BatchNorm3d(nb_neurons_out),
            nn.ReLU(),
            #nn.SELU(),
            nn.MaxPool3d(kernel_size=ksize_pool, stride=strd_pool, padding=pad_pool))
        return layer

    def forward(self, x):
        #print("input:",x.size())
        out = self.layer1(x)
        #print("1st layer", out.size())
        out = self.layer2(out)
        #print("2nd layer", out.size())
        out = self.layer3(out)
        #print("3rd layer", out.size())
        #out = self.layer4(out)
        #print("4th layer", out.size())
        out = out.view(out.size(0), -1)  # Flatten for fully connected layers
        #print("view", out.size())
        out = self.fc(out)
        #print("FC", out.size())
        
        return out