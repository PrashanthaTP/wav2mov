import torch
from torch import nn
from wav2mov.models.layers.conv_layers import Conv1dBlock
from wav2mov.logger import get_module_level_logger
from wav2mov.models.utils import squeeze_batch_frames,get_same_padding

logger = get_module_level_logger(__name__)

class AudioEnocoder(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
        padding_31 = get_same_padding(3,1)
        padding_32 = get_same_padding(3,2)
        padding_42 = get_same_padding(4,2)
        self.conv_encoder = nn.Sequential(
            Conv1dBlock(1,64,330,30,padding=0,use_norm=True,use_act=True,act=nn.ReLU()),#101
            
            Conv1dBlock(64,128,3,1,padding=0,use_norm=True,use_act=True,act=nn.ReLU()),    #98
            Conv1dBlock(128,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv1dBlock(128,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            
            Conv1dBlock(128,256,4,2,padding=padding_42,use_norm=True,use_act=True,act=nn.ReLU()),  #((98-4+2)/2 + 1) = 49
            Conv1dBlock(256,256,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv1dBlock(256,256,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
           
            Conv1dBlock(256,512,3,2,padding=padding_32,use_norm=True,use_act=True,act=nn.ReLU()),  # 49-3+2/2 + 1 = 25
            Conv1dBlock(512,512,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv1dBlock(512,512,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            )
        self.features_len = 25*512#out of conv layers
        self.num_layers = 1
        self.hidden_size = self.hparams['latent_dim_audio'] 
        self.gru = nn.GRU(input_size=self.features_len,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True)            
        self.final_act = nn.Tanh()
        
    def forward(self, x):
        """ x : audio frames of shape B,T,666*5"""
        batch_size,num_frames,_ = x.shape
        x = self.conv_encoder(squeeze_batch_frames(x).unsqueeze(1))
        x = x.reshape(batch_size,num_frames,self.features_len)
        x,_ = self.gru(x)
        return self.final_act(x)  # shape (batch_size,num_frames,hidden_size=latent_dim_audio)