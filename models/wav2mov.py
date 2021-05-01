"""
Supports BATCHING  
audio windowing with coarticulation factor and lr scheduling(new version)
"""
import random

import torch
from wav2mov.core.models.template import TemplateModel
from wav2mov.models.wav2mov_template import Wav2MovTemplate

from wav2mov.core.data.utils import AudioUtil
# from torchvision import transforms as vtransforms
# def process_video(video,hparams):
#     img_channels = hparams['img_channels']
#     img_size = hparams['img_size']
    
#     transforms = get_transforms((img_size, img_size), img_channels)
#     bsize, frames, channels, height, width = video.shape
#     video = video.reshape(bsize*frames, channels, height, width)#vtransforms.Reshape requires input to be of 3d shape
#     video = transforms(video)
#     video = video.reshape(bsize, frames, channels,height,width)
#     return video


class Wav2Mov(TemplateModel):
    def __init__(self,hparams,config,logger):
        super().__init__()
        self.model = Wav2MovTemplate(hparams,config,logger)
        self.hparams = hparams
        self.config = config
        self.logger = logger
        self.set_device()
        
    def set_device(self):
        device = self.hparams['device']
        if device == 'cuda':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
    def forward(self,*args,**kwargs):
        return self.model(*args,**kwargs)

    
    def __get_stride(self):
        return self.hparams['data']['audio_sf']//self.hparams['data']['video_fps']
    
    def on_train_start(self,state):
        self.model.set_train_mode()
        self.to(self.device)
        self.accumulation_steps = self.hparams['data']['batch_size']//self.hparams['data']['mini_batch_size']
        self.zero_grad(set_to_none=True)
        self.audio_util = AudioUtil(self.hparams['data']['coarticulation_factor'],self.__get_stride(),device=self.hparams['device']) 

    def save(self,*args,**kwargs):
        self.model.save(*args,**kwargs)
        
    def load(self,*args,**kwargs):
        self.model.load(*args,**kwargs)
        
    def to_device(self,*args):
        return [arg.to(self.device) for arg in args]
    
    def reset_input(self):
        self.audio = None
        self.audio_frames = None
        self.video = None
        self.fake_video_frames = None
    
    def on_batch_start(self,state):
        self.reset_input()
        
    def setup_input(self,batch,state):
        audio,audio_frames,video = batch
        audio,audio_frames,video = self.to_device(audio,audio_frames,video)
        # video = process_video(video,self.hparams) 
        self.audio = audio
        self.audio_frames = audio_frames
        self.video = video
        
    def squeeze_frames(self,item):
        batch_size,num_frames,*extra = item.shape
        return item.reshape(batch_size*num_frames,*extra)
        
    def get_ref_frames(self,video):
        REF_FRAME_IDX = self.hparams['ref_frame_idx']
        # batch_size,num_frames,channels,height,width = video.shape
        num_frames = video.shape[1]
        ref_frames = video[:,REF_FRAME_IDX,:,:,:]#shape is B,F,C,H,W
        ref_frames = ref_frames.repeat(1,num_frames,1,1,1)
        # ref_frames = ref_frames.reshape(batch_size*num_frames,channels,height,width)
        return ref_frames
    
    def add_fake_frames(self,frames,target_shape):
        batch_size,num_frames,*extra = target_shape
        frames = frames.reshape(batch_size,num_frames,*extra)#from B*F,C,H,W to B,F,C,H,W
        fake_frames = [self.fake_video_frames,frames] if self.fake_video_frames is not None else [frames]
        self.fake_video_frames = torch.cat(fake_frames,dim=1)

    def swap_channel_frame_axes(self,video):
      return video.permute(0,2,1,3,4)#B,F,C,H,W to B,C,F,H,W

    def get_sub_seq(self):
        """ return smaller set of frames from audio,real_video_frames,fake_video_frames"""
           
        ret = {}
      

        # self.video = self.swap_channel_frame_axes(self.video)
   
        OFFSET_SYNC_OUT_OF_SYNC = 15
        NUM_FRAMES_FOR_SYNC = 5
        NUM_FRAMES_ACTUAL = self.video.shape[1] #num_frames is present in 3rd dimension now.(B,C,F,H,W)
        NUM_FRAMES_REQ_MIN =  OFFSET_SYNC_OUT_OF_SYNC + 2*NUM_FRAMES_FOR_SYNC
        if NUM_FRAMES_ACTUAL < NUM_FRAMES_REQ_MIN: 
             raise ValueError(f'Given video should atleast have {NUM_FRAMES_REQ_MIN} frames. Instead given video has {NUM_FRAMES_ACTUAL} frames')

        randpos = random.randint(0, self.video.shape[1]-NUM_FRAMES_FOR_SYNC-OFFSET_SYNC_OUT_OF_SYNC)
      
        real_video_frames = self.video[:, randpos:randpos+NUM_FRAMES_FOR_SYNC, ...]

        audio_frames = self.audio_frames[:,randpos:randpos+NUM_FRAMES_FOR_SYNC,:]
        ref_frames = self.get_ref_frames(real_video_frames)
        self.fake_video_frames = self(audio_frames,ref_frames)
        ret['fake_video_frames'] = self.swap_channel_frame_axes(self.fake_video_frames)
        ret['real_video_frames'] = self.swap_channel_frame_axes(real_video_frames)
        ret['audio_seq'] = self.audio_util.get_limited_audio(self.audio,NUM_FRAMES_FOR_SYNC,start_frame=randpos)
        ret['audio_seq_out_of_sync'] = self.audio_util.get_limited_audio(self.audio,NUM_FRAMES_FOR_SYNC,start_frame=randpos+OFFSET_SYNC_OUT_OF_SYNC)
        
        return ret
    
    def __get_sub_batch(self,fraction=2):
        """creates and yields subbatches of (1/fraction) of num_frames

        Args:
            fraction (int, optional): the denominator of the fraction of frames per subbatch. Defaults to 2.

        Yields:
            sub_batch
        """
        num_frames = self.video.shape[1]
        start_fraction = 0
        for i in range(fraction):
            end_fraction = int((1/fraction)*(i+1)*num_frames) if i!=fraction-1 else num_frames
            batch = {}
            # self.logger.debug(f'inside sub batching with fraction {fraction} of {num_frames} frames| {start_fraction} : {end_fraction}')
            real_video_frames = self.video[:,start_fraction:end_fraction,:,:,:]
            batch['real_video_frames'] = real_video_frames
            batch['ref_video_frames'] = self.get_ref_frames(real_video_frames)
            # self.logger.debug(f'inside get sub batch {ref_video_frames.shape},{real_video_frames.shape}')
            # batch['ref_video_frames'] = self.squeeze_frames(ref_video_frames)
            
            # audio_frames = self.squeeze_frames(audio_frames)
            audio_frames = self.audio_frames[:,start_fraction:end_fraction,:]
            batch['fake_video_frames'] = self(audio_frames,batch['ref_video_frames']) 

            self.add_fake_frames(batch['fake_video_frames'],target_shape=real_video_frames.shape)
            
            start_fraction = end_fraction
            yield batch
            
    def __optimize(self,exclude_fake):     
        losses = {'gen':(0.0,0),
                  'id':(0.0,0),
                  'l1':(0.0,0),
                  'sync':(0.0,0),
                  'seq':(0.0,0)}       
        for sub_batch in self.__get_sub_batch(fraction=2):
            self.model.set_input(sub_batch)
            loss_id = self.model.optimize_id()
            for name,(loss,n) in loss_id.items():
                prev_loss,prev_n = losses.get(name,(0.0,0))
                loss_id[name] = (prev_loss+loss,prev_n+n)
        
        self.model.set_input(self.get_sub_seq())
        loss_sync = self.model.optimize_sync(exclude_fake)
        losses = {**losses,**loss_id,**loss_sync}
        return losses
             
    def optimize(self,state):
        epoch = state.epoch
        batch_idx = state.epoch
        losses = self.__optimize(exclude_fake=epoch<self.hparams['pre_learning_epochs'])
        if (batch_idx+1)%self.accumulation_steps:
            self.model.step()
        return losses


