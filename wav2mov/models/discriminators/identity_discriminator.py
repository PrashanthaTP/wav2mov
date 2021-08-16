import torch 
from torch import nn,optim
from torch.nn import functional as F

from wav2mov.core.models.base_model import BaseModel
from wav2mov.models.utils import squeeze_batch_frames,get_same_padding
from wav2mov.models.layers.conv_layers import Conv2dBlock

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)

class IdentityDiscriminator(BaseModel):
    def __init__(self,hparams):
    
        super().__init__()
        self.hparams = hparams
        in_channels = hparams['in_channels']*2
        chs = self.hparams['chs']
        chs = [in_channels] + chs 
        padding = get_same_padding(kernel_size=4,stride=2)
        relu_neg_slope = self.hparams['relu_neg_slope']
        self.conv_blocks = nn.ModuleList(Conv2dBlock(chs[i],chs[i+1],
                                                     kernel_size=(4,4),
                                                     stride=2,
                                                     padding=padding,
                                                     use_norm=True,
                                                     use_act=True,
                                                     act=nn.LeakyReLU(relu_neg_slope)
                                                     ) for i in range(len(chs)-2)
                                         )

        self.conv_blocks.append(Conv2dBlock(chs[-2],chs[-1],
                                            kernel_size=(4,4),
                                            stride=2,
                                            padding=padding,
                                            use_norm=False,
                                            use_act=False
                                            )
                                )

    def forward(self,x,y):
        """
        x : frame image (B,F,H,W)
        y : still image
        """
        assert x.shape==y.shape

        if len(x.shape)>4:#frame dim present
            x = squeeze_batch_frames(x)
            y = squeeze_batch_frames(y)
        
        x = torch.cat([x,y],dim=1)#along channels
        for block in self.conv_blocks:
          x = block(x)
        return x
    
    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))
