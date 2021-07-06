import torch
from torch import nn,optim

from wav2mov.models.generator.audio_encoder import AudioEnocoder
from wav2mov.models.generator.noise_encoder import NoiseEncoder 
from wav2mov.models.generator.id_encoder import IdEncoder
from wav2mov.models.generator.id_decoder import IdDecoder

from wav2mov.models.utils import squeeze_batch_frames
from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)

class Generator(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams 
        self.id_encoder = IdEncoder(self.hparams)
        self.id_decoder = IdDecoder(self.hparams)
        self.audio_encoder = AudioEnocoder(self.hparams)
        self.noise_encoder = NoiseEncoder(self.hparams)

    def forward(self,audio_frames,ref_frames):
        batch_size,num_frames,*_ = ref_frames.shape
        assert num_frames == audio_frames.shape[1]
        encoded_id , intermediates = self.id_encoder(squeeze_batch_frames(ref_frames))#encoded id is from 1024=>1 
        encoded_id = encoded_id.reshape(batch_size*num_frames,-1,1,1)
        encoded_audio = self.audio_encoder(audio_frames).reshape(batch_size*num_frames,-1,1,1)
        encoded_noise = self.noise_encoder(batch_size,num_frames).reshape(batch_size*num_frames,-1,1,1)
        # logger.debug(f'encoded_id {encoded_id.shape} encoded_audio {encoded_audio.shape} encoded_noise {encoded_noise.shape}')
        encoded = torch.cat([encoded_id,encoded_audio,encoded_noise],dim=1)#along channel dimension
        gen_frames =  self.id_decoder(encoded,intermediates)
        _,*img_shape = gen_frames.shape
        return gen_frames.reshape(batch_size,num_frames,*img_shape)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5, 0.999))