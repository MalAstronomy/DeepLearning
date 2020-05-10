import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__()
        #self.n_channels_in = n_channels_in
        self.lin1 = nn.Linear(n_channels_in**3, 512, bias=True) 
        self.lin2 = nn.Linear(512, 256, bias=True)
        self.lin3 = nn.Linear(256, n_channels_out, bias=True)
        

    def forward(self, xb):
        x = xb.view(-1,50**3) 
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)
        
        return r 