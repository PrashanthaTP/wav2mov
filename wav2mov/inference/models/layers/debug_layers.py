from torch import nn

from inference.utils import get_module_level_logger
logger = get_module_level_logger(__name__)

class IdentityDebugLayer(nn.Module):
    def __init__(self,name):
        super().__init__()
        self.name = name
    def forward(self,x):
        logger.debug(f'{self.name} : {x.shape}')
        return x
