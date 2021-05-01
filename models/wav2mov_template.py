
import os
import torch
from torch.cuda import amp
from torch.optim.lr_scheduler import StepLR

from wav2mov.core.models.template import TemplateModel
from wav2mov.models.generator import Generator, GeneratorBW
from wav2mov.models.sequence_discriminator import SequenceDiscriminator, SequenceDiscriminatorCNN
from wav2mov.models.identity_discriminator import IdentityDiscriminator
from wav2mov.models.patch_disc import PatchDiscriminator
from wav2mov.models.sync_discriminator import SyncDiscriminator
from wav2mov.models.utils import init_net
from wav2mov.losses import GANLoss,SyncLoss,L1_Loss


class Wav2MovTemplate(TemplateModel):
    def __init__(self,hparams,config,logger):
        super().__init__()
        self.hparams = hparams
        self.config = config
        self.logger = logger
        self.accumulation_steps = self.hparams['data']['batch_size']//self.hparams['data']['mini_batch_size']
        self.set_device()

        self.init_models()
        self.init_optims()
        self.init_obj_functions()
        self.init_schedulers() 
        self.scaler = amp.GradScaler()
    
    def set_device(self):
        device = self.hparams['device']
        if device == 'cuda':
            device = 'cpu' if not torch.cuda.is_available() else device
        self.device = torch.device(device)

    def init_models(self):
        self.gen = GeneratorBW(self.hparams['gen'])
        self.seq_disc = SequenceDiscriminatorCNN(self.hparams['disc']['sequence_disc_cnn'])
        self.id_disc = PatchDiscriminator(self.hparams['disc']['patch_disc'])
        # self.id_disc = IdentityDiscriminator(hparams['disc']['identity_disc'])
        self.sync_disc = SyncDiscriminator(self.hparams['disc']['sync_disc'])
        init_net(self.gen)
        init_net(self.seq_disc)
        init_net(self.id_disc)
        init_net(self.sync_disc)
        self.set_train_mode()
        
    def set_train_mode(self):
        self.gen.train()
        self.seq_disc.train()
        self.id_disc.train()
        self.sync_disc.train()

    def init_optims(self):
        self.optim_gen = self.gen.get_optimizer()
        self.optim_seq_disc = self.seq_disc.get_optimizer()
        self.optim_id_disc = self.id_disc.get_optimizer()
        self.optim_sync_disc = self.sync_disc.get_optimizer()

    def init_obj_functions(self):
        self.criterion_gan = GANLoss(self.device)
        self.criterion_L1 = L1_Loss()
        self.criterion_sync = SyncLoss(self.device)

    def init_schedulers(self):
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
            
    def forward(self,audio_frames,ref_video_frames):
        # self.logger.debug(f'inside wav2movtemplate forward {audio_frames.shape} {ref_video_frames.shape}')
        return self.gen(audio_frames,ref_video_frames)
    
    def set_input(self,batch:dict):
        # self.audio_frames = batch.get('audio_frames')
        self.audio_seq = batch.get('audio_seq')#used by sync
        self.ref_video_frames = batch.get('ref_video_frames')#used by id
        self.real_video_frames = batch.get('real_video_frames')#usef by id,sync,seq
        self.fake_video_frames = batch.get('fake_video_frames')#used by id,sync,seq
        self.audio_seq_out_of_sync = batch.get('audio_seq_out_of_sync')#used by sync
        
        ##########################
        # for key,value in batch:
        #     self.logger.debug(f'{key} : {value.shape} ')
        
    def backward_id(self):
        with amp.autocast():
            disc_out = self.id_disc(self.real_video_frames,self.ref_video_frames)
            loss_id = self.criterion_gan(disc_out,is_real_target=True)/2
            disc_out = self.id_disc(self.fake_video_frames.detach(),
                                    self.ref_video_frames)
            loss_id += self.criterion_gan(disc_out, is_real_target=False)/2
            loss_ret = loss_id.item()
            loss_id /= self.accumulation_steps
        self.scaler.scale(loss_id).backward()
        return {'id':(loss_ret,self.real_video_frames.shape[0])}

    def backward_sync(self,exclude_fake=False):
        scale =  2 if exclude_fake else 3
        with amp.autocast():
            disc_out = self.sync_disc(self.audio_seq,self.real_video_frames)
            
            loss_sync = self.criterion_sync(*disc_out,
                                           is_real_target=True)/scale
            
            disc_out = self.sync_disc(self.audio_seq_out_of_sync,self.real_video_frames)
    
            loss_sync += self.criterion_sync(*disc_out,
                                            is_real_target=False)/scale
            
            if not exclude_fake: 
                disc_out = self.sync_disc(self.audio_seq,self.fake_video_frames.detach())
                loss_sync += self.criterion_sync(*disc_out,
                                                is_real_target=False)/scale
                
            loss_ret = loss_sync.item()
            loss_sync /= self.accumulation_steps
        self.scaler.scale(loss_sync).backward()
        return {'sync':(loss_ret,self.audio_seq.shape[0])}

    def backward_seq(self,exclude_fake=False):
        scale = 1 if exclude_fake else 2
        with amp.autocast():
            disc_out = self.seq_disc(self.real_video_frames)
            loss_seq = self.criterion_gan(disc_out, is_real_target=True)/scale
            if not exclude_fake:
                disc_out = self.seq_disc(self.fake_video_frames)
                loss_seq += self.criterion_gan(disc_out, is_real_target=False)/scale
            ret_loss = loss_seq.item()
            loss_seq /= self.accumulation_steps
        self.scaler.scale(loss_seq).backward()
        return {'seq':(ret_loss,self.real_video_frames.shape[0])}

    def backward_gen_id(self):
        with amp.autocast():
         
            ##################################
            # ID discriminator
            ##################################
            id_disc_out = self.id_disc(self.fake_video_frames,self.ref_video_frames)

            loss_gen = self.criterion_gan(id_disc_out,
                                           is_real_target=True) * self.hparams['scales']['lambda_id_disc']

            ##################################
            # L1 Criterion
            ##################################
            loss_l1 = self.criterion_L1(self.fake_video_frames,
                                        self.real_video_frames)*self.hparams['scales']['lambda_L1']
            loss_ret = {'gen':(loss_gen.item(),self.fake_video_frames.shape[0]),
                        'l1':(loss_l1.item(),self.fake_video_frames.shape[0])}
            loss_gen += loss_l1
            
            
            loss_gen /= self.accumulation_steps
        self.scaler.scale(loss_gen).backward()
        return loss_ret

    def backward_gen_sync(self):
        with amp.autocast():
            ##################################
            # SYNC discriminator
            ##################################
            sync_disc_out = self.sync_disc(self.audio_seq,self.fake_video_frames)

            loss_gen = self.criterion_sync(*sync_disc_out,
                                          is_real_target=True)*self.hparams['scales']['lambda_sync_disc']

            loss_ret = loss_gen.item()
            loss_gen /= self.accumulation_steps
        self.scaler.scale(loss_gen).backward()
        return {'gen' : (loss_ret,self.audio_seq.shape[0])}

    def backward_gen_seq(self):
        with amp.autocast():
            seq_disc_out = self.seq_disc(self.fake_video_frames)
            
            loss_gen = self.criterion_gan(seq_disc_out,
                                          is_real_target=True)*self.hparams['scales']['lambda_seq_disc']
            loss_ret = loss_gen.item()
            loss_gen /= self.accumulation_steps
        self.scaler.scale(loss_gen).backward()
        return {'gen':(loss_ret,self.fake_video_frames.shape[0])}
    
    def step(self):
        self.scaler.step(self.optim_id_disc)
        self.scaler.step(self.optim_sync_disc)
        self.scaler.step(self.optim_gen)
        # self.scaler.step(self.optim_seq_disc)
        
        self.optim_gen.zero_grad(set_to_none=True)
        self.optim_id_disc.zero_grad(set_to_none=True)
        self.optim_sync_disc.zero_grad(set_to_none=True)
        # self.optim_seq_disc.zero_grad(set_to_none=True)
      
        self.scaler.update()
        
    def optimize_id(self):
        losses = {}
        losses_id = self.backward_id()
        losses_gen = self.backward_gen_id()
        losses = {**losses_id,**losses_gen}
        return losses
    
    def optimize_sync(self,exclude_fake):
        losses = {}
        losses = {**losses,**self.backward_sync(exclude_fake)}
        if exclude_fake:
            return losses
        losses = {**losses,**self.backward_gen_sync()}
        return losses
            
    def optimize_seq(self,exclude_fake):
        losses = {}
        losses = {**losses,**self.backward_seq(exclude_fake)}
        if exclude_fake:
            return losses
        losses = {**losses,**self.backward_gen_seq()}
        return losses
        
    
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
     