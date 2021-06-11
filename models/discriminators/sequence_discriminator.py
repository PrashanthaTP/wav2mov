import torch
from torch import nn,optim

from wav2mov.core.models.base_model import BaseModel
from wav2mov.models.layers.conv_layers import Conv2dBlock
from wav2mov.models.utils import get_same_padding,squeeze_batch_frames

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)

class SequenceDiscriminator(BaseModel):
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams 
        in_size, h_size, num_layers = self.hparams['in_size'],self.hparams['h_size'],self.hparams['num_layers']
        self.gru = nn.GRU(input_size=in_size,hidden_size=h_size,num_layers=num_layers,batch_first = True)
        in_channels = self.hparams['in_channels']
        chs = [in_channels] + self.hparams['chs']
        kernel,stride = 4,2
        padding = get_same_padding(kernel,stride)
        cnn = nn.ModuleList([Conv2dBlock(chs[i],chs[i+1],kernel,stride,padding,
                                         use_norm=True,use_act=True,
                                         act=nn.LeakyReLU(0.01)) for i in range(len(chs)-2)])
        cnn.append(Conv2dBlock(chs[-2],chs[-1],kernel,stride,padding,
                               use_norm=False,use_act=True,
                               act=nn.Tanh()))
        self.cnn = nn.Sequential(*cnn)
        ############################################
        # channels : 3  => 64 => 128 => 256 => 512          
        # frame sz : 256=> 128 => 64  =>  32 => 16 =>8         
        ############################################

    def forward(self,frames):
        """frames : B,T,C,H,W"""
        img_height = frames.shape[-2]
        frames = frames[...,0:img_height//2,:]#consider upper half
        batch_size,num_frames,*img_size = frames.shape
        frames = squeeze_batch_frames(frames)
        frames = self.cnn(frames)
        frames = frames.reshape(batch_size,num_frames,-1)
        out,_ = self.gru(frames)#out is of shape (batch_size,seq_len,num_dir*hidden_dim)
        return out[:,-1,:]#batch_size,hidden_size

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))
