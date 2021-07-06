import random
import torch 
from torch import nn 

class GANLoss(nn.Module):
    """ To abstract away the task of creating real/fake labels and calculating loss
    [Reference]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """    
    def __init__(self,device,real_label=None,fake_label=0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        # self.register_buffer('real_label',torch.tensor(real_label))
        # self.register_buffer('fake_label',torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()
        self.device = device 
        
    def get_target_tensor(self,preds,is_real_target):
        real_label = torch.tensor( round(random.uniform(0.8,1),2) if self.real_label is None else self.real_label) 
        target_tensor  = real_label if is_real_target else torch.tensor(self.fake_label)
        return target_tensor.expand_as(preds).to(self.device)
    
    def forward(self,preds,is_real_target):
        target_tensor = self.get_target_tensor(preds,is_real_target)
        return self.loss(preds,target_tensor)    
