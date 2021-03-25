from abc import abstractmethod
import torch 
from torch import nn 

import logging 
logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def save_to(self,checkpoint_fullpath):
        torch.save(self,checkpoint_fullpath)
        logger.log(f'Model saved at {checkpoint_fullpath}','INFO')
    
    def load_from(self,checkpoint_fullpath):
        try:
             self.load_statedict(torch.load(checkpoint_fullpath))
        except:
            logger.log(f'Cannot load checkpoint from {checkpoint_fullpath}',type="ERROR")
        
    @abstractmethod
    def forward(self,*args):
        raise NotImplementedError(f'Forward method is not defined in {self.__class__.__name__}')
    
