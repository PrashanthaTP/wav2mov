"""
Supports BATCHING  
audio windowing with coarticulation factor and lr scheduling(new version)
"""
import random
import torch

from wav2mov.core.models import TemplateModel
from wav2mov.core.data.utils import AudioUtil

from wav2mov.models.wav2mov_template import Wav2MovTemplate

class Wav2MovTrainer(TemplateModel):
    def __init__(self,hparams,config,logger):
        super().__init__()
        self.model = Wav2MovTemplate(hparams,config,logger)
        self.hparams = hparams
        self.config = config
        self.logger = logger
        self.train_sync_expert = True
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
        audio_sf = self.hparams['data']['audio_sf']
        coarticulation_factor = self.hparams['data']['coarticulation_factor']
        self.audio_util = AudioUtil(audio_sf,coarticulation_factor,self.__get_stride(),device=self.hparams['device']) 

    def save(self,*args,**kwargs):
        self.model.save(*args,**kwargs)
        
    def load(self,*args,**kwargs):
        ret = self.model.load(*args,**kwargs)
        return ret
        
    def to_device(self,*args):
        return [arg.to(self.device) for arg in args]
    
    def clear_input(self):
        self.audio = None
        self.audio_frames = None
        self.video = None
        self.fake_video_frames = None
        self.model.clear_input()
        
    def on_epoch_start(self,state):
        if state.epoch == self.hparams['pre_learning_epochs']:
          self.logger.debug(f'============================== Adversarial traininig with id disc and sync disc starts now ================================')

    def on_batch_start(self,state):
        self.clear_input()

    def setup_input(self,batch,state):
        audio,audio_frames,video = batch
        audio,audio_frames,video = self.to_device(audio,audio_frames,video)
        # video = process_video(video,self.hparams) 
        self.audio = audio
        self.audio_frames = audio_frames
        self.video = video
        self.logger.debug(f'input {video.shape}')
        
    def squeeze_frames(self,item):
        batch_size,num_frames,*extra = item.shape
        return item.reshape(batch_size*num_frames,*extra)
        
    def __get_random_indices(self,low,high,num_frames):
        return random.sample(range(low,high),k=num_frames)
        # return torch.randint(low,high,(num_frames,))  #size must be tuple

    def get_ref_frames(self,video):
        # REF_FRAME_IDX = self.hparams['ref_frame_idx']
        # self.logger.debug(f'ref {video.shape}')
        batch_size,num_frames,*_ = video.shape
        ############################ METHOD 1 #############################
        # for each video , for each frame , a random frame from the same video will be the reference frame
        #############################################################
        # ref_frame_indices = self.__get_random_indices(0,num_frames,num_frames)    
        # ref_frames = video[:,ref_frame_indices]
        ############################ METHOD 2 ##################################
        # ref_frames = video[torch.arange(video.shape[0]),ref_frame_indices]
        # ref_frames = ref_frames.unsqueeze(1)#B,1,C,H,W   
        # ref_frames = ref_frames.repeat(1,num_frames,1,1,1)
        #########################################################################
        REF_FRAME_IDX = int(random.randint(0,1)*(num_frames-1))  
        ref_frames = video[:,REF_FRAME_IDX,:,:,:]
        ref_frames = ref_frames.unsqueeze(1)#B,1,C,H,W   
        ref_frames = ref_frames.repeat(1,num_frames,1,1,1)
        
        return ref_frames
    
    def add_fake_frames(self,frames):
        fake_frames = [self.fake_video_frames,frames] if self.fake_video_frames is not None else [frames]
        self.fake_video_frames = torch.cat(fake_frames,dim=1)
    
    def re_gen(self,fraction):
        # self.logger.debug(f'line 503 fake_video_frames : {self.fake_video_frames.shape}')
        self.fake_video_frames = None
        # self.logger.debug(f'line 475 fake_video_frames cleared ')
        num_frames = self.video.shape[1]
        start_fraction = 0
        for i in range(fraction):
            end_fraction = int((1/fraction)*(i+1)*num_frames) if i!=fraction-1 else num_frames
            batch = {}
            # self.logger.debug(f'inside sub batching with fraction {fraction} of {num_frames} frames| {start_fraction} : {end_fraction}')
            batch['real_video_frames'] = self.video[:,start_fraction:end_fraction,:,:,:]
            batch['ref_video_frames'] = self.get_ref_frames(batch['real_video_frames'])
      
            audio_frames = self.audio_frames[:,start_fraction:end_fraction,:]
            batch['fake_video_frames'] = self(audio_frames,batch['ref_video_frames']) 
            self.add_fake_frames(batch['fake_video_frames'])
            
            start_fraction = end_fraction

    def get_sub_seq(self,for_sync_disc):
        """ return smaller set of frames from audio,real_video_frames,fake_video_frames"""
        ret = {}

        OFFSET_SYNC_OUT_OF_SYNC = 15
        NUM_FRAMES_ACTUAL = self.video.shape[1] #num_frames is present in 2nd dimension now.(B,F,C,H,W)
        NUM_FRAMES_FOR_SYNC = 5 if for_sync_disc else 15

        if for_sync_disc:
          NUM_FRAMES_REQ_MIN =  OFFSET_SYNC_OUT_OF_SYNC + 2*NUM_FRAMES_FOR_SYNC 
          if NUM_FRAMES_ACTUAL < NUM_FRAMES_REQ_MIN: 
              raise ValueError(f'Given video should atleast have {NUM_FRAMES_REQ_MIN} frames. Instead given video has {NUM_FRAMES_ACTUAL} frames')

        randpos_limit = self.video.shape[1]-NUM_FRAMES_FOR_SYNC
        if for_sync_disc:
            randpos_limit -= OFFSET_SYNC_OUT_OF_SYNC
            
        randpos = random.randint(0,randpos_limit)
        real_video_frames = self.video[:, randpos:randpos+NUM_FRAMES_FOR_SYNC, ...]
        audio_frames = self.audio_frames[:,randpos:randpos+NUM_FRAMES_FOR_SYNC,:]

        ref_frames = self.get_ref_frames(real_video_frames)
        self.fake_video_frames = self(audio_frames,ref_frames)
 
        ret['fake_video_frames'] = self.fake_video_frames
        ret['real_video_frames'] = real_video_frames 
     
        if not for_sync_disc:
          return ret

        ret['audio_seq'] = self.audio_util.get_limited_audio(self.audio,NUM_FRAMES_FOR_SYNC,start_frame=randpos,get_mfccs=True)
        ret['audio_seq_out_of_sync'] = self.audio_util.get_limited_audio(self.audio,NUM_FRAMES_FOR_SYNC,start_frame=randpos+OFFSET_SYNC_OUT_OF_SYNC,get_mfccs=True)
        
        return ret
           
    def __get_sub_batch(self,fraction=2):
        """creates and yields subbatches of (1/fraction) of num_frames

        Args:
            fraction (int, optional): the denominator of the fraction of frames per subbatch. Defaults to 2.

        Yields:
            sub_batch
        """
        # self.logger.debug(f'sub batch {self.video.shape}')
        num_frames = self.video.shape[1]
        start_fraction = 0
        for i in range(fraction):
            end_fraction = int((1/fraction)*(i+1)*num_frames) if i!=fraction-1 else num_frames
            batch = {}
            # self.logger.debug(f'inside sub batching with fraction {fraction} of {num_frames} frames| {start_fraction} : {end_fraction} | {end_fraction-start_fraction}')
            real_video_frames = self.video[:,start_fraction:end_fraction,:,:,:]
            batch['real_video_frames'] = real_video_frames
            batch['ref_video_frames'] = self.get_ref_frames(real_video_frames)
            audio_frames = self.audio_frames[:,start_fraction:end_fraction,:]
            batch['fake_video_frames'] = self(audio_frames,batch['ref_video_frames'])
            self.add_fake_frames(batch['fake_video_frames'])
            start_fraction = end_fraction
            yield batch
            
    def __optimize(self,adversarial):     
        self.fake_video_frames = None
        num_frames_fraction = self.hparams['num_frames_fraction']
        losses = {'gen':(0.0,0),
                  'id':(0.0,0),
                  'l1':(0.0,0),
                  'sync':(0.0,0),
                  'seq':(0.0,0)}       
          
        for sub_batch in self.__get_sub_batch(fraction=num_frames_fraction):
            self.model.set_input(sub_batch)
            loss_id_dict = self.model.optimize_id(adversarial=adversarial,scale=num_frames_fraction)
            losses = self._merge_losses(losses,loss_id_dict)
        
        self.model.step_id_disc()
        ##############################################
        if adversarial and  self.train_sync_expert:#at the end of prelearning
            self.train_sync_expert = False
            self.model.freeze_sync_disc()
            # self.model.freeze_seq_disc()
        ##############################################
      
        self.model.set_input(self.get_sub_seq(for_sync_disc=True))
        loss_sync_dict = self.model.optimize_sync(adversarial)
        if not adversarial:
            self.model.step_sync_disc()

        self.model.set_input(self.get_sub_seq(for_sync_disc=False))#if not generated again , the computation graph would not be available
        loss_seq_dict = self.model.optimize_seq(adversarial=True)
        self.model.step_seq_disc()

        self.clear_input()#clears even the input video
        losses = self._merge_losses(losses,loss_sync_dict,loss_seq_dict)
     
        self.model.step_gen()
        self.model.update_scale()
        
        return losses

    def on_epoch_end(self,state):
        # self.model.update_learning_rate(state.epoch)
        pass
    
    def _merge_losses(self,*loss_dicts):
        merged_losses = {}
        for loss_dict in loss_dicts:
            for key,(loss,n) in loss_dict.items():
                merged_losses[key] = merged_losses.get(key,(0.0,0))
                prev_loss,prev_n = merged_losses[key]
                merged_losses[key] = (prev_loss+loss,prev_n+n)
        return merged_losses

    def optimize(self,state):
        epoch = state.epoch
        losses = self.__optimize(adversarial=((epoch)>=self.hparams['pre_learning_epochs']))
        # batch_idx = state.epoch
        # if (batch_idx+1)%self.accumulation_steps:
        #     self.model.step()
        return losses

    def on_run_end(self,state):
      self.logger.debug(str(self))
      