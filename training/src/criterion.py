import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self,yhat,y):
        return self.mse(yhat,y)
    
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self,yhat,y):
        return self.ce(yhat,y)
