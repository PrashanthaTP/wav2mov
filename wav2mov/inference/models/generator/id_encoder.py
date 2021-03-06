import torch
from torch import nn

from inference.models.layers.conv_layers import Conv2dBlock
from inference.models.utils import get_same_padding
from inference.utils import get_module_level_logger

logger = get_module_level_logger(__name__)

class IdEncoder(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
        in_channels = hparams['in_channels']
        chs = self.hparams['chs']
        chs = [in_channels] + chs +[1]# 1 is added here not in params because see how channels are being used in  id_decoder 
        padding = get_same_padding(kernel_size=4,stride=2)
        self.conv_blocks = nn.ModuleList(Conv2dBlock(chs[i],chs[i+1],
                                                     kernel_size=(4,4),
                                                     stride=2,
                                                     padding=padding,
                                                     use_norm=True,
                                                     use_act=True,
                                                     act=nn.ReLU()
                                                     ) for i in range(len(chs)-2)
                                         )

        self.conv_blocks.append(Conv2dBlock(chs[-2],chs[-1],
                                            kernel_size=(4,4),
                                            stride=2,
                                            padding=padding,
                                            use_norm=False,
                                            use_act=True,
                                            act=nn.Tanh()))
    def forward(self,images):
        intermediates = []
        for block in self.conv_blocks[:-1]:
            images = block(images)
            intermediates.append(images)
        encoded = self.conv_blocks[-1](images)
        return encoded,intermediates
        