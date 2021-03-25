import torch
from torch import nn

class L1_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    def forward(self,preds,targets):
        """
        preds and targets are of the shape = (N,C,H,W)
        """
        # height = preds.shape[-2]
        # preds = preds[...,height//2:,:]
        # targets = targets[...,height//2:,:]
        # # print(torch.mean(abs(preds-targets)).shape) #torch.size([])
        # return torch.mean(abs(preds-targets))#get mean of error of all batch samples
        
        return self.loss(preds,targets)