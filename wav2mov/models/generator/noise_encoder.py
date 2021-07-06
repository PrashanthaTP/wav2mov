import torch
from torch import nn

from wav2mov.logger import get_module_level_logger
from wav2mov.models.utils import squeeze_batch_frames

logger = get_module_level_logger(__name__)

class NoiseEncoder(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
        self.features_len = 10
        self.hidden_size = self.hparams['latent_dim_noise']
        self.gru = nn.GRU(input_size=self.features_len,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True)
        #input should be of shape batch_size,seq_len,input_size
    def forward(self,batch_size,num_frames):
        noise = torch.randn(batch_size,num_frames,self.features_len,device=self.hparams['device'])
        out,_ = self.gru(noise)
        return out#(batch_size,seq_len,hidden_size)
