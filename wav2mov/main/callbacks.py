from collections import defaultdict
import os
import time
import torch
from wav2mov.core.engine.callbacks import Callbacks

from wav2mov.logger import TensorLogger
from wav2mov.utils.misc import AverageMetersList,ProgressMeter

from wav2mov.logger import get_module_level_logger
m_logger = get_module_level_logger(__name__)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TensorBoardCallback(Callbacks):

    def __init__(self,options,config):
        self.options = options
        self.config = config
        self.tensor_logger = self.get_tensor_logger()
        
    def on_batch_end(self,batch_idx,epoch,steps,logs):
        self.add_to_board( logs['batch_loss'], steps, scalar_type='loss')
        
    def setup_tensor_logger(self,config):
        tensor_logger = TensorLogger(config['runs_dir'])
        writer_names = ['writer_gen', 'writer_sync_disc',
                        'writer_seq_disc', 'writer_id_disc']
        tensor_logger.add_writers(writer_names)
        return tensor_logger

    def get_tensor_logger(self):
        return self.setup_tensor_logger(self.config) if self.options.log in ['y', 'yes'] else None


    def add_to_board(self, losses, global_step, scalar_type):
        for name, value in losses.items():
            writer_name = 'writer_' + name
            self.tensor_logger.add_scalar(writer_name, scalar_type+'_'+name, value, global_step)

class LoggingCallback(Callbacks):
    def __init__(self,options,hparams,config,logger) :
        self.options = options
        self.hparams = hparams
        self.config = config
        self.logger = logger
        
    def on_run_start(self,state):
        self.logger.info(f'[RUN] version {self.config.version}')
        accumulation_steps = self.hparams['data']['batch_size']//self.hparams['data']['mini_batch_size']
        num_batches = state.num_batches//accumulation_steps
        self.logger.debug(f'[DATALOADER] num_videos {self.options.num_videos}')
        self.logger.debug(f'[DATALOADER] min_batch_size {self.hparams["data"]["mini_batch_size"]}')
        self.logger.debug(f'[DATALOADER] batch_size {self.hparams["data"]["batch_size"]}')
        self.logger.debug(f'[DATALOADER] total batches {num_batches}')
        self.logger.debug(f'[RUN] num_epochs : {self.hparams["num_epochs"]} | pre_learning_epochs : {self.hparams["pre_learning_epochs"]}')
        self.logger.debug(f'[RUN] adversarial_with_id : {self.hparams["adversarial_with_id"]} |adversarial_with_sync: {self.hparams["adversarial_with_sync"]}| adversarial_with_seq : {self.hparams["stop_adversarial_with_sync"]}')
        self.logger.debug(f'[RUN] starting epoch : {state.start_epoch}')

    def on_run_end(self,state):
        self.logger.debug(f'[Run] version {self.config.version}')

class LossMetersCallback(Callbacks):
    def __init__(self,options,hparams,config,logger,verbose=True):
        self.options = options
        self.hparams = hparams
        self.config = config
        self.logger = logger
        self.verbose = verbose
        self.accumulation_steps = self.hparams['data']['batch_size']//self.hparams['data']['mini_batch_size']
        
        self.losses = defaultdict(list)

        # m_logger.warning(f'checking m_logger {m_logger.name} {m_logger.level}')
        
    def on_train_start(self,state):
        self.batch_loss_meter = AverageMetersList(('id',
                                                   'sync',
                                                   'seq', 
                                                   'gen',
                                                   'l1'),
                                                    fmt=':0.4f') 
             
        self.epoch_loss_meter = AverageMetersList(('id',
                                                   'sync',
                                                   'seq', 
                                                   'gen',
                                                   'l1'),
                                                    fmt=':0.4f') 
        if self.verbose:
            # print(f'sfjskdfj {state}')
            num_batches = state.num_batches//self.accumulation_steps
            num_epochs = self.hparams['num_epochs']
            self.batch_progress_meter = ProgressMeter(num_batches,self.batch_loss_meter.as_list(),'[BATCH]')
            self.epoch_progress_meter = ProgressMeter(num_epochs,self.epoch_loss_meter.as_list(),'[EPOCH]')

    def on_batch_start(self,state):
        pass
    
    def on_epoch_start(self,state):
        self.batch_loss_meter.reset()
        self.epoch_loss_meter.reset()
        
    def on_batch_end(self,state):
        batch_idx = state.batch_idx
        batch_size = state.cur_batch_size
        if (batch_idx+1)%self.accumulation_steps == 0:
            self.batch_loss_meter.update(state.logs)
            batch_avg_loss = self.batch_loss_meter.average()
            self.epoch_loss_meter.update(self.update_with_multiplier(batch_avg_loss,batch_size))
            if self.verbose:
                self.logger.info(self.batch_progress_meter.get_display_str((batch_idx+1)//self.accumulation_steps))
                self.logger.debug(f"{bcolors.BOLD}{bcolors.OKCYAN}batch_idx={batch_idx+1} | num_batches={state.num_batches} | epoch={state.epoch}|num_epochs={self.hparams['num_epochs']} {bcolors.ENDC}")
            self.batch_loss_meter.reset()

    def on_epoch_end(self,state):
        epoch = state.epoch
        losses = self.epoch_loss_meter.average()#name,value

        for name,value in losses.items():
          self.losses[name].append(value)
        # self.logger.debug(self.losses)
        self.logger.debug(losses)
        # raise ValueError('stop')

        # losses = {name : (value,1) for name,value in losses}
        # losses = self.update_with_multiplier(losses,multiplier=1)
        # self.epoch_loss_meter.update(losses)
        if self.verbose:
            m_logger.debug("="*25)
            self.logger.info(self.epoch_progress_meter.get_display_str(epoch+1))
            m_logger.debug("="*25)
        
        if epoch % 5 == 0:
            self.save_losses(state)

    def save_losses(self,state):
        self.losses['epochs'] = [state.start_epoch,state.epoch]
        DIR = os.path.join(self.config['runs_dir'],'losses')
        os.makedirs(DIR,exist_ok=True)
        path = os.path.join(DIR,f'losses_{self.config.version}.pt')
        torch.save(self.losses,path)
        self.logger.debug(f'Loss values are saved : {path}')

    def on_run_end(self,state):
        # self.losses['epochs'] = [state.start_epoch,state.epoch]
        # DIR = os.path.join(self.config['runs_dir'],'losses')
        # os.makedirs(DIR,exist_ok=True)
        # path = os.path.join(DIR,f'losses_{self.config.version}.pt')
        # torch.save(self.losses,path)
        # self.logger.debug(f'Loss values are saved : {path}')
        self.save_losses(state)
        
    def update_with_multiplier(self,losses,multiplier):
        updated = {}
        for name,value in losses.items():
            if isinstance(value,tuple):
                updated[name] = value
            else:
                updated[name] = (value,multiplier)
        return updated

class TimeTrackerCallback(Callbacks):
    def __init__(self,hparams,logger):
        self.accumulation_steps = hparams['data']['batch_size']//hparams['data']['mini_batch_size']
        self.logger = logger
        self.train_duration = 0.0
        
    def on_train_start(self,state):
        self.batch_duration = 0.0
        self.epoch_duration = 0.0
        self.num_batches = state.num_batches//self.accumulation_steps
        
    def on_epoch_start(self,state):
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()
    
    def on_batch_end(self,state):
        batch_idx = state.batch_idx
        if (batch_idx+1)%self.accumulation_steps==0:
            duration_sec = (time.time()-self.batch_start_time)
            duration_min = duration_sec/60
            self.logger.debug(f'[BATCH_TIME] batch {(batch_idx+1)//self.accumulation_steps} took {duration_min:0.2f} minutes or {duration_sec:0.2f} seconds.')
            self.batch_start_time = time.time()
            
    def on_epoch_end(self, state):
        epoch = state.epoch
        duration_sec = time.time()-self.epoch_start_time
        duration_min = duration_sec/60
        self.logger.debug(f'[EPOCH_TIME] epoch {epoch+1} took {duration_min:0.2f} minutes or {duration_sec:0.2f} seconds.')
        self.train_duration += duration_sec
    
    def on_train_end(self, state):
        self.logger.debug(f'[TRAIN_TIME] training took {self.train_duration/60:0.2f} minutes or {self.train_duration:0.2f} seconds.')
        

class ModelCheckpoint(Callbacks):
    def __init__(self,model,hparams,config,logger,save_every):
        self.model = model
        self.hparams = hparams
        self.config = config
        self.logger = logger
        self.are_hparams_saved = False
        self.save_every = save_every
        
    def on_epoch_end(self, state):
        epoch = state.epoch
        if (epoch+1)%self.save_every==0:
            self.model.save(epoch)
            if not self.are_hparams_saved:
                #to avoid multiple file writes as hparams not edited anywhere after initial setup
                self.hparams.save(self.config['params_checkpoint_fullpath'])
                self.logger.debug(f'hparams saved.')
                self.are_hparams_saved = True
            self.logger.debug(f'Model saved.')
                
    def on_train_end(self,state):
        self.model.save(state.epoch)
