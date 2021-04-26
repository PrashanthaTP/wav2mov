import os

from wav2mov.core.engine import TemplateEngine
from wav2mov.core.engine import CallbackDispatcher,Callbacks,CallbackEvents as Events


class State:
    def __init__(self,names):
        self.names = names
        for name in self.names:
            setattr(self,name,None)
            
    def reset(self,*names):
        for name in names:
            setattr(self,name,None)
            
        
class Engine(TemplateEngine):
    def __init__(self,logger):
        super().__init__()
        self.logger = logger
        self.state = State(['num_batches','epoch','batch_idx','start_epoch','logs'])
    
        
    def configure(self,hparams,options,config):
        self.hparams = hparams
        self.options = options
        self.config = config
    
    def on_run_start(self,model,train_dl):
        #log 
        # accumulation_steps = self.hparams['data']['batch_size']//self.hparams['data']['mini_batch_size']
        # num_batches = len(train_dl)//accumulation_steps
        # self.logger.info('options num_videos : {self.options.num_videos}')
        # self.logger.info(f'train_dl : len(train_dl) :{len(train_dl)} : num_batches: {num_batches}')
       pass 
    
    def resume_checkpoint(self,model):
        if getattr(self.options, 'model_path', None) is  None:
            return
        loading_version = os.path.basename(self.options.model_path)
        self.logger.debug(f'Loading pretrained weights : {self.config.version} <== {loading_version}')

        prev_epoch = model.load(checkpoint_dir=self.options.model_path)
        if prev_epoch is not None:
            self.state.start_epoch = prev_epoch+1
            self.logger.debug(f'Start Epoch : {prev_epoch+1}')
        self.logger.debug(f'weights loaded successfully: {self.config.version} <== {loading_version}')
    
    def dispatch(self, event):
        super().dispatch(event,state=self.state)
        
    def run(self,model,dataloaders_ntuple,callbacks=None):
        callbacks = callbacks or []
        callbacks = [model] + callbacks
        self.register(callbacks)
        self.resume_checkpoint(model)
        
        train_dl = dataloaders_ntuple.train
        self.state.num_batches = len(train_dl)
        num_epochs = self.hparams['num_epochs']
        self.dispatch(Events.RUN_START)
        self.dispatch(Events.TRAIN_START)
        for epoch in range(self.state.start_epoch,num_epochs):
            self.state.epoch = epoch
            self.dispatch(Events.EPOCH_START)
            for batch_idx,batch in enumerate(train_dl):
                self.state.batch_idx = batch_idx
                self.dispatch(Events.BATCH_START)
                model.set_input(batch,state=self.state)
                logs = model.optimize(state=self.state)
                self.state.logs = logs
                self.dispatch(Events.BATCH_END)
                self.state.reset(['logs'])
                
            self.dispatch(Events.EPOCH_END)
        self.dispatch(Events.TRAIN_END)
        self.dispatch(Events.RUN_END)
        
                
        
        

    