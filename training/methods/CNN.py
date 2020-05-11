import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_channels_in, n_channels_out):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(n_channels_in, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=1))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128*5**3, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, n_channels_out),
            nn.ReLU())    

    def forward(self, x):
        print("input:",x.size())
        out = self.layer1(x)
        print("1st layer", out.size())
        out = self.layer2(out)
        print("2nd layer", out.size())
        out = self.layer3(out)
        print("3rd layer", out.size())
        out = out.reshape(out.size(0), -1)  
        print("reshape", out.size())
        out = self.fc(out)
        print("FC", out.size())
        
        #out = self.sig(out)
        #out = nn.functional.sigmoid(out)
        #out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
        return out