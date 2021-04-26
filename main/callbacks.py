import time
from wav2mov.core.engine.callbacks import Callbacks

from wav2mov.logger import TensorLogger
from wav2mov.utils.misc import AverageMetersList,ProgressMeter

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
            
class LossMetersCallback(Callbacks):
    def __init__(self,options,hparams,logger,verbose=True):
        self.options = options
        self.hparams = hparams
        self.logger = logger
        self.verbose = verbose
        self.accumulation_steps = self.hparams['batch_size']//self.hparams['mini_batch_size']
        
    def on_train_start(self,state):
        self.batch_loss_meter = AverageMetersList(('id_disc',
                                                   'sync_disc',
                                                   'seq_disc', 
                                                   'gen',
                                                   'l1'),
                                                    fmt=':0.4f') 
             
        self.epoch_loss_meter = AverageMetersList(('id_disc',
                                                   'sync_disc',
                                                   'seq_disc', 
                                                   'gen',
                                                   'l1'),
                                                    fmt=':0.4f') 
        if self.verbose:
            num_batches = state.num_batches//self.accumulation_steps
            num_epochs = self.hparams['num_epochs']
            self.batch_progress_meter = ProgressMeter(num_batches,self.batch_loss_meter,'[BATCH]')
            self.epoch_progress_meter = ProgressMeter(num_epochs,self.epoch_loss_meter,'[EPOCH]')

    def on_batch_start(self,state):
        pass
    
    def on_epoch_start(self,state):
        self.batch_loss_meter.reset()
        self.epoch_loss_meter.reset()
        
    def on_batch_end(self,state):
        batch_idx = state.batch_idx
        if (batch_idx+1)%self.accumulation_steps == 0:
            self.batch_loss_meter.update(state.logs)
            self.epoch_loss_meter.update(self.batch_loss_meter.average())
            if self.verbose:
                self.logger.info(self.batch_progress_meter.get_display_str(batch_idx+1))
                
    def on_epoch_end(self,state):
        epoch = state.epoch
        losses = self.batch_loss_meter.average()#name,value
        losses = {name : (value,1) for name,value in losses}
        self.epoch_loss_meter.update(losses)
        if self.verbose:
            self.logger.info(self.epoch_progress_meter.get_display_str(epoch+1))
    


class TimeTrackerCallback(Callbacks):
    def __init__(self,hparams,logger):
        self.accumulation_steps = hparams['batch_size']//self.hparams['mini_batch_size']
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
        if batch_idx+1==self.num_batches:
            duration_sec = (time.time()-self.batch_start_time)
            duration_min = duration_sec//60
            self.logger.debug(f'[BATCH_TIME] batch {batch_idx+1} took {duration_min:0.2f} minutes or {duration_sec:0.2f} seconds.')
            
    def on_epoch_end(self, state):
        epoch = state.epoch
        duration_sec = time.time()-self.epoch_duration
        duration_min = duration_sec//60
        self.logger.debug(f'[EPOCH_TIME] epoch {epoch+1} took {duration_min:0.2f} minutes or {duration_sec:0.2f} seconds.')
        self.train_duration += duration_sec
    
    def on_train_end(self, state):
        self.logger.debug(f'[TRAIN_TIME] training took {self.train_duration//60:0.2f} minutes or {self.train_duration:0.2f} seconds.')
        

class ModelCheckpoint(Callbacks):
    def __init__(self,model,hparams,save_every):
        self.model = model
        self.hparams = hparams
        self.are_hparams_saved = False
        self.save_every = save_every
        
    def on_epoch_end(self, state):
        epoch = state.epoch
        if (epoch+1)%self.save_every==0:
            self.model.save(epoch)
            if not self.are_hparams_saved:
                #to avoid multiple file writes as hparams not edited anywhere after initial setup
                self.hparams.save()
                self.are_hparams_saved = True
                
    def on_train_end(self,state):
        self.model.save(state.epoch)