import torch
from torch.autograd.grad_mode import no_grad
from wav2mov.core.models.template import TemplateModel
from wav2mov.core.data.utils import AudioUtil

from wav2mov.models.generator import Generator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def no_grad_wrapper(fn):
    def wrapper(*args,**kwargs):
        with torch.no_grad():
            return fn(*args,**kwargs)
    return wrapper

class Wav2movInferencer(TemplateModel):
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
        self.device = 'cpu'
        self.gen = Generator(hparams['gen'])
        self.stride = self.hparams['data']['audio_sf']//self.hparams['data']['video_fps']
        self.audio_util = AudioUtil(self.hparams['data']['coarticulation_factor'],self.stride,self.device)        
    def load(self,checkpoint):
        self.gen.load_state_dict(checkpoint)
    def _squeeze_batch_frames(self,target):
        batch_size,num_frames,*extra = target.shape
        return target.reshape(batch_size*num_frames,*extra)

    def forward(self,audio_frames,ref_video_frames):
        self.gen.eval() 
        batch_size,num_frames,*extra = ref_video_frames
        assert batch_size==audio_frames.shape[0] and num_frames ==audio_frames.shape[1]
        # audio_frames = self._squeeze_batch_frames(audio_frames)
        # ref_video_frames = self._squeeze_batch_frames(ref_video_frames)
        fake_frames =  self.gen(audio_frames,ref_video_frames)
        return fake_frames

    @no_grad_wrapper
    def generate(self,audio_frames,ref_video_frames,fraction=None):
        if fraction is None:
            return self(audio_frames,ref_video_frames)
        else:
            return self._generate_with_fraction(audio_frames,ref_video_frames,fraction)
        
    @no_grad_wrapper
    def test(self,audio_frames,video):
        """test the generation of face images

        Args:
            audio (tensor | (B,S)): 
            audio_frames (tensor | (B,F,Sw)): 
            video (tensor | (B,F,C,H,W)):
        """
        fraction = 10
        ref_video_frames = self._get_ref_video_frames(video)
        return self.generate(audio_frames,ref_video_frames,fraction)
    
    def _get_ref_video_frames(self,video):
        num_frames = video.shape[1]
        REF_FRAME_IDX = int(0.6*num_frames)  
        ref_video_frames = video[:,REF_FRAME_IDX,:,:,:]
        ref_video_frames = ref_video_frames.unsqueeze(1)
        ref_video_frames = ref_video_frames.repeat(1,num_frames,1,1,1)
        return ref_video_frames
    
    def _squeeze_frames(self,video):
        bsize,nframes,*extra = video.shape
        return video.reshape(bsize*nframes,*extra)
    
        
    
    def _generate_with_fraction(self,audio_frames,ref_video_frames,fraction):
        num_frames = audio_frames.shape[1]
        start_frame = 0
        fake_video_frames= []
        for i in range(fraction):
            end_frame = ((1/fraction)*(i+1)*num_frames) if i!=fraction-1 else num_frames
            audio_frames_sample = audio_frames[:,start_frame:end_frame,...]
            ref_video_frames_sample = ref_video_frames[:,start_frame:end_frame,...]
            fake_video_frames.append(self(audio_frames_sample,ref_video_frames_sample))
            
        return torch.cat([fake_video_frames])
        
        