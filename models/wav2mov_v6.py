"""
audio windowing with coarticulation factor and lr scheduling(new version)
"""
import os
import random

import torch
from torch.cuda import amp
from torch.optim.lr_scheduler import StepLR

from wav2mov.core.models.template import TemplateModel
from wav2mov.models.generator_v6 import Generator, GeneratorBW
from wav2mov.models.sequence_discriminator import SequenceDiscriminator, SequenceDiscriminatorCNN
from wav2mov.models.identity_discriminator import IdentityDiscriminator
from wav2mov.models.patch_disc import PatchDiscriminator
from wav2mov.models.sync_discriminator_v6 import SyncDiscriminator
from wav2mov.models.utils import init_net
from wav2mov.losses.gan_loss import GANLoss
from wav2mov.losses.l1_loss import L1_Loss


class Wav2MovBW(TemplateModel):
    def __init__(self, config, hparams, logger):
        super().__init__()
        self.config = config
        self.hparams = hparams
        self.logger = logger

        device = hparams['device']
        if device == 'cuda':
            device = 'cpu' if not torch.cuda.is_available() else device
        self.device = torch.device(device)

        self.gen = GeneratorBW(hparams['gen'])
        self.seq_disc = SequenceDiscriminatorCNN(
            hparams['disc']['sequence_disc_cnn'])
        self.id_disc = PatchDiscriminator(hparams['disc']['patch_disc'])
        # self.id_disc = IdentityDiscriminator(hparams['disc']['identity_disc'])
        self.sync_disc = SyncDiscriminator(hparams['disc']['sync_disc'])

        init_net(self.gen)
        init_net(self.seq_disc)
        init_net(self.id_disc)
        init_net(self.sync_disc)

        self.optim_gen = self.gen.get_optimizer()
        self.optim_seq_disc = self.seq_disc.get_optimizer()
        self.optim_id_disc = self.id_disc.get_optimizer()
        self.optim_sync_disc = self.sync_disc.get_optimizer()

        self.criterion_gan = GANLoss(self.device)
        self.criterion_L1 = L1_Loss()

        gen_step_size = self.hparams['scheduler']['gen']['step_size']
        discs_step_size = self.hparams['scheduler']['discs']['step_size']
        gen_gamma = self.hparams['scheduler']['gen']['gamma']
        discs_gamma = self.hparams['scheduler']['discs']['gamma']
        
        self.scheduler_gen = StepLR(self.optim_gen,step_size=gen_step_size,
                                    gamma=gen_gamma,verbose=True)
        self.scheduler_id_disc = StepLR(self.optim_id_disc,step_size=discs_step_size,
                                        gamma=discs_gamma,verbose=True)
        self.scheduler_sync_disc = StepLR(self.optim_sync_disc,step_size=discs_step_size,
                                          gamma=discs_gamma,verbose=True)
        # self.scheduler_seq_disc = StepLR(self.seq_disc,step_size=discs_step_size,gamma=discs_gamma,verbose=True)

        self.scaler = amp.GradScaler()

    def forward(self):
        # self.prev_fake_video_frame = self.curr_fake_video_frame
        self.curr_fake_video_frame = self.gen( self.curr_real_audio_frame, 
                                              self.still_image)
        # print('gen : train_mode : ',self.gen.training)
        if self.gen.training:
            self.fake_frames = torch.cat([self.fake_frames,
                                          self.curr_fake_video_frame.detach().unsqueeze(2)],
                                         dim=2) \
                if self.fake_frames is not None else self.curr_fake_video_frame.detach().unsqueeze(dim=2)
            # print(f'added to frame to fake_frames : {len(self.fake_frames)},{self.fake_frames.shape}')

    def _set_frame_history(self):
        # self.real_frames = None
        self.fake_frames = None
        # self.audio_frames = None

        self.curr_real_video_frame = None
        self.curr_real_audio_frame = None

        self.curr_fake_video_frame = None

    def on_train_start(self):
        self._set_frame_history()
        self.to(self.device)
        self.gen.train()
        self.id_disc.train()
        self.seq_disc.train()
        self.sync_disc.train()
        self.accumulation_steps = self.hparams['data']['batch_size']//self.hparams['data']['mini_batch_size']
        self.zero_grad(set_to_none=True)
        
    def on_epoch_start(self, epoch):
        self.loss_gen = 0.0
        
    def on_batch_start(self):
        self._set_frame_history()
        
    def set_condition(self, still_image):
        self.still_image = still_image.to(self.device)

    def set_input(self, audio_frame, video_frame):

        self.curr_real_video_frame = video_frame.to(self.device)
        self.curr_real_audio_frame = audio_frame.to(self.device)

    def backward_gen(self):
        """requires <curr_fake_frame> to be populated before hand that is during discriminator training
        """
        with amp.autocast():
         
            ##################################
            # ID discriminator
            ##################################
            id_disc_out = self.id_disc(self.curr_fake_video_frame,
                                       self.still_image)

            loss_gen = self.criterion_gan(id_disc_out,
                                           is_real_target=True) * self.hparams['scales']['lambda_id_disc']

            ##################################
            # L1 Criterion
            ##################################
            loss_l1 = self.criterion_L1(self.curr_fake_video_frame,
                                          self.curr_real_video_frame)*self.hparams['scales']['lambda_L1']
            loss_ret = {'gen':loss_gen.item(),'l1':loss_l1.item()}
            loss_gen += loss_l1
            
            
            loss_gen /= self.accumulation_steps
        self.scaler.scale(loss_gen).backward()
        return loss_ret

    def backward_gen_seq(self,audio_frames):

        with amp.autocast():
            ##################################
            # SYNC discriminator
            ##################################
            sync_disc_out = self.sync_disc(audio_frames,
                                           self.fake_frames)

            loss_gen = self.criterion_gan(sync_disc_out,
                                          is_real_target=True)*self.hparams['scales']['lambda_sync_disc']

            # seq_disc_out = self.seq_disc(self.fake_frames)
            
            # loss_gen = self.criterion_gan(seq_disc_out,
            #                               is_real_target=True)*self.hparams['scales']['lambda_seq_disc']
        # self.loss_gen += loss_gen
            loss_ret = loss_gen.item()
            loss_gen /= self.accumulation_steps
        self.scaler.scale(loss_gen).backward()
        return loss_ret

   

    def backward_seq_disc(self, real_frames):
        with amp.autocast():
            disc_out = self.seq_disc(real_frames)
            loss_seq = self.criterion_gan(disc_out, is_real_target=True)/2
            disc_out = self.seq_disc(self.fake_frames)
            loss_seq += self.criterion_gan(disc_out, is_real_target=False)/2
            ret_loss = loss_seq.item()
            loss_seq /= self.accumulation_steps
        self.scaler.scale(loss_seq).backward()
        return ret_loss

    def backward_sync_disc(self,audio_frames,real_frames,out_of_sync_audio_frames):
        with amp.autocast():
            disc_out = self.sync_disc(audio_frames,real_frames)
            loss_sync = self.criterion_gan(disc_out,
                                           is_real_target=True)/3

            disc_out = self.sync_disc(audio_frames,self.fake_frames.detach())
            loss_sync += self.criterion_gan(disc_out,
                                            is_real_target=False)/3
            
            disc_out = self.sync_disc(out_of_sync_audio_frames,real_frames)
            loss_sync += self.criterion_gan(disc_out,
                                            is_real_target=False)/3
            
            loss_ret = loss_sync.item()
            loss_sync /= self.accumulation_steps
        self.scaler.scale(loss_sync).backward()
        return loss_ret

    def backward_id_disc(self):
        with amp.autocast():
            disc_out = self.id_disc(self.curr_real_video_frame,
                                    self.still_image)
            loss_id = self.criterion_gan(disc_out, is_real_target=True)/2

            disc_out = self.id_disc(self.curr_fake_video_frame.detach(),
                                    self.still_image)
            loss_id += self.criterion_gan(disc_out, is_real_target=False)/2
            loss_ret = loss_id.item()
            loss_id /= self.accumulation_steps
        self.scaler.scale(loss_id).backward()
        return loss_ret

    def scheduler_step(self):
      self.scheduler_gen.step()
      self.scheduler_id_disc.step()
    #   self.scheduler_sync_disc.step()
      # self.scheduler_seq_disc.step()

    def step(self):
        #! self.zero_grad applies to all the optimizers
        # self.step_optimizers()
        self.scaler.step(self.optim_id_disc)
        self.scaler.step(self.optim_gen)
        # self.scaler.step(self.optim_sync_disc)
        # self.scaler.step(self.optim_seq_disc)
        # self.scheduler_step()
        
        self.optim_gen.zero_grad()
        self.optim_id_disc.zero_grad()
        # self.optim_sync_disc.zero_grad()
        # self.optim_seq_disc.zero_grad()
        # self.update_scalers()
        self.scaler.update()
        
    def optimize_parameters(self):
        losses = {}
        with amp.autocast():
            self.forward()  # generate fake frame
        # accumulate lossees
        # losses['sync_disc'] = self.backward_sync_disc()
        losses['id_disc'] = self.backward_id_disc()
        losses_gen = self.backward_gen()
        losses = {**losses,**losses_gen}
        return losses

    def on_epoch_end(self, epoch):
        if epoch <= self.hparams['scheduler']['max_epoch']:
            self.scheduler_step()

        
    def batch_descent(self):
        self.step()
        
    def optimize_sequence(self, real_frames,get_audio_frames_from_range):
        ############
        # real_frames has the shape (B,F,C,H,W)
        # make it (B,C,F,H,W) #think of 3d cnn
        #############
        real_frames = real_frames.permute(0, 2, 1, 3, 4)
        self.logger.debug(f"[POSSIBLE BUG] :(frame accumulation) "
                        f"fake_frames.shape : {self.fake_frames.shape} and \n"
                        f"real_frames.shape : {real_frames.shape}")
        losses = {}
        NUM_FRAMES = 5
        # 2nd pos shape has num_frames : F
        randpos = random.randint(0, real_frames.shape[2]-NUM_FRAMES-10)
        real_frames = real_frames[..., randpos:randpos+NUM_FRAMES, :, :]
        self.fake_frames = self.fake_frames[...,randpos:randpos+NUM_FRAMES, :, :]
        audio_frames = get_audio_frames_from_range(randpos, randpos+NUM_FRAMES-1)
        out_of_sync_audio_frames = get_audio_frames_from_range(randpos+10, randpos+NUM_FRAMES+10-1)
        self.logger.debug(f"audio_frames.shape : {audio_frames.shape} and \n"
                          f"out_of_sync_audio_frames.shape : {out_of_sync_audio_frames.shape}")
        # accumulate losses
        # losses['seq_disc'] = self.backward_seq_disc(real_frames)
        losses['sync_disc'] = self.backward_sync_disc(audio_frames,real_frames,out_of_sync_audio_frames)
        losses['gen'] = self.backward_gen_seq(audio_frames)
        self._set_frame_history()
        return losses



    def has_attribute(self,attr):
        return hasattr(self,attr)


    def save_state_dict(self,name,checkpoint,**kwargs):
        entity = getattr(self,name)
        torch.save({'state_dict':entity.state_dict(),**kwargs},checkpoint)
  
    
     
    def load_state_dict(self,name,checkpoint):
        entity = getattr(self,name)
        entity.load_state_dict(torch.load(checkpoint,map_location=self.hparams['device'])['state_dict'])
        """
        If map_location not provided then
        #!RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
        """

    def save_optimizers_and_schedulers(self):
        models = ['gen','id_disc','sync_disc','seq_disc']
        for model in models:
            optim_name = f'optim_{model}'
            scheduler_name = f'scheduler_{model}'
            if hasattr(self,optim_name):
                optim_checkpoint = self.config[f'optim_{model}_checkpoint_fullpath']
                self.save_state_dict(optim_name,checkpoint=optim_checkpoint)
            if hasattr(self,scheduler_name):
                scheduler_checkpoint = self.config[f'scheduler_{model}_checkpoint_fullpath']
                self.save_state_dict(scheduler_name,checkpoint=scheduler_checkpoint)


    def load_optimizers_and_schedulers(self,checkpoint_dir):
        version = os.path.basename(checkpoint_dir)
        pt_file = checkpoint_dir + os.sep + '%(model)s_'+version+'.pt'
      
        models = ['gen', 'id_disc', 'sync_disc', 'seq_disc']
        loaded = []
        for model in models:
            optim_name = f'optim_{model}'
            scheduler_name = f'scheduler_{model}'
            if hasattr(self, f'optim_{model}'):
                optim_checkpoint = pt_file % {'model': optim_name}
                self.load_state_dict(optim_name, optim_checkpoint)
                loaded.append(optim_name)
            if hasattr(self, f'scheduler_{model}'):
                scheduler_checkpoint = pt_file % {'model': scheduler_name}
                self.load_state_dict(scheduler_name, scheduler_checkpoint)
                loaded.append(scheduler_name)

        self.logger.debug(f'[LOAD] loaded successfully {loaded}')


    def save(self, epoch=0):
        torch.save({'state_dict': self.gen.state_dict(), 'epoch': epoch},
                   self.config['gen_checkpoint_fullpath'])
        torch.save({'state_dict': self.seq_disc.state_dict(), 'epoch': epoch},
                   self.config['seq_disc_checkpoint_fullpath'])
        torch.save({'state_dict': self.sync_disc.state_dict(), 'epoch': epoch},
                   self.config['sync_disc_checkpoint_fullpath'])
        torch.save({'state_dict': self.id_disc.state_dict(), 'epoch': epoch},
                   self.config['id_disc_checkpoint_fullpath'])

        self.save_optimizers_and_schedulers()

    def load(self, checkpoint_dir):
        self.to(self.hparams['device'])
        version = os.path.basename(checkpoint_dir)
        pt_file = checkpoint_dir + os.sep + '%(model_name)s_'+version+'.pt'
        try:
            self.gen.load_state_dict(torch.load(pt_file % {'model_name': 'gen'},map_location=self.hparams['device'])['state_dict'])
            self.sync_disc.load_state_dict(torch.load( pt_file % {'model_name': 'sync_disc'},map_location=self.hparams['device'])['state_dict'])
            self.seq_disc.load_state_dict(torch.load( pt_file % {'model_name': 'seq_disc'},map_location=self.hparams['device'])['state_dict'])
            self.id_disc.load_state_dict(torch.load(pt_file % {'model_name': 'id_disc'},map_location=self.hparams['device'])['state_dict'])
            
            self.load_optimizers_and_schedulers(checkpoint_dir)
            
            return torch.load(pt_file % {'model_name': 'gen'})['epoch']
        except Exception as e:
            self.logger.exception(e)
