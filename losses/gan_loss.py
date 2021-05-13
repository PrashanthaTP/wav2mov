import random
import torch 
from torch import nn 

class GANLoss(nn.Module):
    """ To abstract away the task of creating real/fake labels and calculating loss
    [Reference]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    
    """
    
    def __init__(self,device,real_label=None,fake_label=0.0):
        super().__init__()
        if real_label is None:
            real_label = round(random.uniform(0.8,1),2)
        self.register_buffer('real_label',torch.tensor(real_label))
        self.register_buffer('fake_label',torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()
        self.device = device 
        
    def get_target_tensor(self,preds,is_real_target):
        target_tensor = self.real_label if is_real_target else self.fake_label
        return target_tensor.expand_as(preds).to(self.device)
    
    def forward(self,preds,is_real_target):
        target_tensor = self.get_target_tensor(preds,is_real_target)
        return self.loss(preds,target_tensor)    
