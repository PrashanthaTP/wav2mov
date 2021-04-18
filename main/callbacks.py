import time
from wav2mov.core.engine.callbacks_v2 import Callbacks

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
            
class BatchMetricsCallback(Callbacks):
    def __init__(self,options,hparams):
        self.options = options
        self.hparams = hparams
        self.batch_loss_meter = AverageMetersList(('id_disc',
                                                   'sync_disc',
                                                   'seq_disc', 
                                                   'gen',
                                                   'l1'),
                                                    fmt=':0.4f') 
     
    def on_batch_start(self,batch_idx):
        pass
    
    def on_batch_end(self,batch_idx,logs):
        loss = logs['batch_loss']
        
class EpochMetricsCallback:
    def __init__(self,options,hparams):
        self.options = options
        self.hparams = hparams
        self.epoch_loss_meter = AverageMetersList(('id_disc',
                                                   'sync_disc',
                                                   'seq_disc', 
                                                   'gen',
                                                   'l1'),
                                                    fmt=':0.4f') 
        
