"""Wav2Mov Model Version 1
+ id disc : Patch GAN : [input] : generated/real video frame conditioned on original still image
+ sync disc : [input] : audio frame and corresponding generated/real video frame
+ sequence disc : [input] : 20 consecutive generated/real video frames 
+  Training : 
    + optimization in terms of how real frames are passed to seq disc
    + seq disc gets updated after all the frames of a video
    + use of 4 GradScalers
+ Results : 
    + 
"""
import os
import random

import torch
from torch import nn
from torch.cuda import amp

from wav2mov.core.models.template import TemplateModel
from wav2mov.models.generator import Generator, GeneratorBW
from wav2mov.models.sequence_discriminator import SequenceDiscriminator, SequenceDiscriminatorCNN
from wav2mov.models.identity_discriminator import IdentityDiscriminator
from wav2mov.models.patch_disc import PatchDiscriminator
from wav2mov.models.sync_discriminator import SyncDiscriminator
from wav2mov.models.utils import init_net
from wav2mov.losses.gan_loss import GANLoss
from wav2mov.losses.l1_loss import L1_Loss


class Wav2Mov(nn.Module):
    def __init__(self, config, hparams, logger):
        super().__init__()
        self.config = config
        self.hparams = hparams
        self.logger = logger

        self.gen = Generator(hparams['gen'])
        self.seq_disc = SequenceDiscriminator(hparams['disc']['sequence'])
        self.id_disc = PatchDiscriminator(hparams['disc']['patch_disc'])
        # self.id_disc = IdentityDiscriminator(hparams['disc']['identity'])
        self.sync_disc = SyncDiscriminator(hparams['disc']['sync'])

     
        
    def forward(self, speech, face_image):
        pass


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
        self.seq_disc = SequenceDiscriminatorCNN(hparams['disc']['sequence_disc_cnn'])
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

        self.scaler_disc = amp.GradScaler()
        self.scaler_gen = amp.GradScaler()
        self.scaler_seq = amp.GradScaler()
        self.scaler_gen_seq = amp.GradScaler()
        # self.scaler = amp.GradScaler()

    def forward(self):
        # self.prev_fake_video_frame = self.curr_fake_video_frame
        self.curr_fake_video_frame = self.gen(
            self.curr_real_audio_frame, self.still_image)
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
        
    def on_epoch_start(self,epoch):
        self.loss_gen = 0.0
    def on_batch_start(self):
        self._set_frame_history()
    
    def set_condition(self, still_image):
        self.still_image = still_image.to(self.device)
        # self.num_frames = num_frames
        
    def set_input(self, audio_frame, video_frame):

        self.curr_real_video_frame = video_frame.to(self.device)
        self.curr_real_audio_frame = audio_frame.to(self.device)
        # self.real_frames = torch.cat([self.real_frames, self.curr_real_video_frame.detach().unsqueeze(2)], dim=2) \
        #     if self.real_frames is not None else self.curr_real_video_frame.detach().unsqueeze(dim=2)
    
   
    def backward_gen(self):
        """requires <curr_fake_frame> to be populated before hand that is during discriminator training
        """
        with amp.autocast():
            ##################################
            # SYNC discriminator 
            ##################################
            sync_disc_out = self.sync_disc(self.curr_real_audio_frame,
                                            self.curr_fake_video_frame)

            loss_gen = self.criterion_gan(sync_disc_out,
                                            is_real_target=True)*self.hparams['scales']['lambda_sync_disc']

            ##################################
            # ID discriminator 
            ##################################
            id_disc_out = self.id_disc(self.curr_fake_video_frame,
                                        self.still_image)

            loss_gen += self.criterion_gan(id_disc_out,
                                            is_real_target=True) * self.hparams['scales']['lambda_id_disc']

            ##################################
            # L1 Criterion
            ##################################
            loss_gen += self.criterion_L1(self.curr_fake_video_frame,
                                          self.curr_real_video_frame)*self.hparams['scales']['lambda_L1']
      
        self.scaler_gen.scale(loss_gen).backward()
        return loss_gen.item()

    def backward_gen_seq(self):

        with amp.autocast():
            seq_disc_out = self.seq_disc(self.fake_frames)
            loss_gen = self.criterion_gan(seq_disc_out,
                                          is_real_target=True)*self.hparams['scales']['lambda_seq_disc']
        # self.loss_gen += loss_gen
        self.scaler_gen_seq.scale(loss_gen).backward()
        return loss_gen.item()
    
    def backward_gen_v2(self,curr_real_video_frame,curr_fake_video_frame,curr_real_audio_frame):
        
        with amp.autocast():
          
            
            sync_disc_out = self.sync_disc(curr_real_audio_frame,
                                            curr_fake_video_frame)

            loss_gen = self.criterion_gan(sync_disc_out,
                                            is_real_target=True)*self.hparams['scales']['lambda_sync_disc']

            id_disc_out = self.id_disc(curr_fake_video_frame,
                                        self.still_image)

            loss_gen += self.criterion_gan(id_disc_out,
                                            is_real_target=True) * self.hparams['scales']['lambda_id_disc']

            loss_gen += self.criterion_L1(curr_fake_video_frame,
                                          curr_real_video_frame)*self.hparams['scales']['lambda_L1']
        # self.loss_gen += loss_gen
        self.scaler_gen.scale(loss_gen).backward()
        return loss_gen.item()


    def backward_seq_disc(self,real_frames):
        with amp.autocast():
            disc_out = self.seq_disc(real_frames)
            loss_seq = self.criterion_gan(disc_out, is_real_target=True)/2
            disc_out = self.seq_disc(self.fake_frames)
            loss_seq += self.criterion_gan(disc_out, is_real_target=False)/2
        
        self.scaler_seq.scale(loss_seq).backward()
        return loss_seq.item()
        
    def backward_sync_disc(self):
        with amp.autocast():
            disc_out = self.sync_disc(self.curr_real_audio_frame, 
                                      self.curr_real_video_frame)
            loss_sync = self.criterion_gan(disc_out, 
                                           is_real_target=True)/2
            
            disc_out = self.sync_disc( self.curr_real_audio_frame, 
                                      self.curr_fake_video_frame.detach())
            loss_sync += self.criterion_gan(disc_out,
                                            is_real_target=False)/2
            
        self.scaler_disc.scale(loss_sync).backward()
        return loss_sync


    def backward_id_disc(self):
        with amp.autocast():
            disc_out = self.id_disc(self.curr_real_video_frame, 
                                    self.still_image)
            loss_id = self.criterion_gan(disc_out, is_real_target=True)/2
            
            disc_out = self.id_disc(self.curr_fake_video_frame.detach(), 
                                    self.still_image)
            loss_id += self.criterion_gan(disc_out, is_real_target=False)/2
            
        self.scaler_disc.scale(loss_id).backward()
        return loss_id.item()

    def optimize_parameters(self):
        losses = {}
        with amp.autocast():
            self.forward()  # generate fake frame

        self.optim_sync_disc.zero_grad()
        losses['sync_disc'] = self.backward_sync_disc()
        self.scaler_disc.step(self.optim_sync_disc)

        self.optim_id_disc.zero_grad()
        losses['id_disc'] = self.backward_id_disc()
        self.scaler_disc.step(self.optim_id_disc)

        self.optim_gen.zero_grad()
        losses['gen'] = self.backward_gen()
        self.scaler_gen.step(self.optim_gen)

        self.scaler_disc.update()
        self.scaler_gen.update()
        
        return losses

    def optimize_sequence(self,real_frames):
        ############
        # real_frames has the shape (B,F,C,H,W)
        # make it (B,C,F,H,W) #think of 3d cnn
        real_frames = real_frames.permute(0,2,1,3,4)
        
        #############
        losses = {}
        NUM_FRAMES = 20
        
        randpos = random.randint(0,real_frames.shape[2]-NUM_FRAMES)#2nd pos shape has num_frames : F
        
        # num_frames = self.real_frames.shape[-3]
        # for i in range(num_frames):
        #     losses['gen'] += self.backward_gen_v2(self.real_frames[...,i,:,:],self.fake_frames[...,i,:,:],self.audio_frames[:,i,:])
        #     break
        self.logger.debug(f'[Possible BUG] real_frames.shape {real_frames.shape} fake_frames.shape {self.fake_frames.shape}')
        real_frames = real_frames[..., randpos:randpos+NUM_FRAMES, :, :]
        self.fake_frames = self.fake_frames[..., randpos:randpos+NUM_FRAMES, :, :]

        self.optim_seq_disc.zero_grad()
        losses['seq_disc'] = self.backward_seq_disc(real_frames)
        self.scaler_seq.step(self.optim_seq_disc)
        self.scaler_seq.update()

        self.optim_gen.zero_grad()
        losses['gen'] = self.backward_gen_seq()
        self.scaler_gen_seq.step(self.optim_gen)
        self.scaler_gen_seq.update()
      
        return losses

    def save(self, epoch=0):
        torch.save({'state_dict': self.gen.state_dict(), 'epoch': epoch},
                   self.config['gen_checkpoint_fullpath'])
        torch.save({'state_dict': self.seq_disc.state_dict(), 'epoch': epoch},
                   self.config['seq_disc_checkpoint_fullpath'])
        torch.save({'state_dict': self.sync_disc.state_dict(), 'epoch': epoch},
                   self.config['sync_disc_checkpoint_fullpath'])
        torch.save({'state_dict': self.id_disc.state_dict(), 'epoch': epoch},
                   self.config['id_disc_checkpoint_fullpath'])

    def load(self, checkpoint_dir):
        checkpoint = os.path.basename(checkpoint_dir)
        pt_file = checkpoint_dir + '\\%(model_name)s_'+checkpoint+'.pt'
        try:
            self.gen.load_state_dict(torch.load(
                pt_file % {'model_name': 'gen'})['state_dict'])
            self.sync_disc.load_state_dict(torch.load(
                pt_file % {'model_name': 'sync_disc'})['state_dict'])
            self.seq_disc.load_state_dict(torch.load(
                pt_file % {'model_name': 'seq_disc'})['state_dict'])
            self.id_disc.load_state_dict(torch.load(
                pt_file % {'model_name': 'id_disc'})['state_dict'])

            return torch.load(pt_file % {'model_name': 'gen'})['epoch']
        except Exception as e:
            self.logger.exception(e)

    
