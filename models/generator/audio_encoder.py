import torch
from torch import nn

from wav2mov.logger import get_module_level_logger
from wav2mov.models.utils import squeeze_batch_frames

logger = get_module_level_logger(__name__)

class AudioEnocoder(nn.Module):
    """ 
       >>> x = x.reshape(x.shape[0], 1, -1)
       >>> x = self.conv(x)
       >>> x = self.fc(x.reshape(x.shape[0],-1))
       >>> return x #shape (batch_size,28*28)
    """

    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
        use_bias = True
        #audio = 666*5=3330
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 330, 30,bias=use_bias),#input 666 output (3330-330+0)/30 + 1 = 101
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, 1,bias=use_bias),#((101-3+0)/1)+1 =98
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1,4,2,1),#((98-4+2)/2)+1 = 48 +1 = 49
            nn.Tanh(),
            )
        self.features_len = 49#out of conv layers
        self.num_layers = 1
        self.hidden_size = self.hparams['latent_dim_audio'] 
        self.gru = nn.GRU(input_size=self.features_len,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True)            
        self.final_act = nn.Tanh()
    def forward(self, x):
        # logger.debug(f'{x.shape}')
        batch_size,num_frames,_ = x.shape
        x = self.conv(squeeze_batch_frames(x).unsqueeze(1))
        # x = self.audio_fc(x.reshape(x.shape[0], -1))
        # logger.debug(f'audio after convs {x.shape}')
        x.reshape(batch_size,num_frames,self.features_len)
        x,_ = self.gru(x)
        #B,seq_len,hidden_size
        return self.final_act(x)  # shape (batch_size,num_frames,hidden_size=latent_dim_audio)