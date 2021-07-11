import torch
from torch import nn

from wav2mov.models.layers.conv_layers import Conv2dBlock,ConvTranspose2dBlock,DoubleConvTranspose2d
from wav2mov.logger import get_module_level_logger
from wav2mov.models.utils import get_same_padding

logger = get_module_level_logger(__name__)

class IdDecoder(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
        self.in_channels = hparams['in_channels']
        self.latent_channels = hparams['latent_dim']
        latent_id_height,latent_id_width = hparams['latent_dim_id']
        chs = self.hparams['chs'][::-1]#1024,512,256,128,64
        chs = [self.latent_channels] + chs + [self.in_channels]
        padding = get_same_padding(kernel_size=4,stride=2)
        self.convs = nn.ModuleList()
        self.convs.append(ConvTranspose2dBlock(in_ch=chs[0],
                                               out_ch=chs[1],
                                               kernel_size=(latent_id_height,latent_id_width),
                                               stride=2,
                                               padding=0,
                                               use_norm=True,
                                               use_act=True,
                                               act=nn.ReLU()
                                               ))
        for i in range(1,len(chs)-2):
            self.convs.append( DoubleConvTranspose2d(in_ch=chs[i],
                                                    skip_ch=chs[i],
                                                    out_ch=chs[i+1],
                                                    kernel_size=(4,4),
                                                    stride=2,
                                                    padding=padding,
                                                    use_norm=True,
                                                    use_act=True,
                                                    act=nn.ReLU())
                             )
        self.convs.append(ConvTranspose2dBlock(in_ch=chs[-2],
                                               out_ch=chs[-1],
                                               kernel_size=(4,4),
                                               stride=2,
                                               padding=padding,
                                               use_norm=True,
                                               use_act=True,
                                               act=nn.Tanh()
                                               ))
    def forward(self,encoded,skip_outs):
        """[summary]

        Args:
            skip_outs ([type]): 64,128,256,512,1024
            encoded ([type]): latent_channels, 
        """
        encoded = encoded.reshape(encoded.shape[0],self.latent_channels,1,1)
        skip_outs = skip_outs[::-1]
        for i,block in enumerate(self.convs[:-2]):
            x = block(encoded)
            encoded = torch.cat([x,skip_outs[i]],dim=1)#cat along channel axis
        
        encoded = self.convs[-2](encoded)
        return self.convs[-1](encoded)
        