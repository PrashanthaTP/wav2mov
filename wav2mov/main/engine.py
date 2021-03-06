import os

from wav2mov.core.engine import TemplateEngine
from wav2mov.core.engine import CallbackEvents as Events


class State:
    def __init__(self,names):
        self.names = names
        for name in self.names:
            setattr(self,name,None)
            
    def reset(self,names):
        for name in names:
            setattr(self,name,None)
            
        
class Engine(TemplateEngine):
    def __init__(self,options,hparams,config,logger):
        super().__init__()
        self.logger = logger
        self.configure(options,hparams,config)
        self.state = State(['num_batches','cur_batch_size','epoch','batch_idx','start_epoch','logs'])
    
        
    def configure(self,options,hparams,config):
        self.hparams = hparams
        self.options = options
        self.config = config
    
    def load_checkpoint(self,model):
        prev_epoch = 0
        if getattr(self.options, 'model_path', None) is  None:
            return prev_epoch
        loading_version = os.path.basename(self.options.model_path)
        self.logger.debug(f'Loading pretrained weights : {self.config.version} <== {loading_version}')

        prev_epoch = model.load(checkpoint_dir=self.options.model_path)
        if prev_epoch is None:
            prev_epoch = 0
        self.logger.debug(f'weights loaded successfully: {self.config.version} <== {loading_version}')
        return prev_epoch + 1
    
    def dispatch(self, event):
        super().dispatch(event,state=self.state)
        
    def run(self,model,dataloaders_ntuple,callbacks=None):
        callbacks = callbacks or []
        callbacks = [model] + callbacks
        self.register(callbacks)
        self.state.start_epoch = self.load_checkpoint(model) 
        
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
                self.state.cur_batch_size = batch[0].shape[0] #makes the system tight coupled though!?
                self.dispatch(Events.BATCH_START)
                model.setup_input(batch,state=self.state)
                logs = model.optimize(state=self.state)
                self.state.logs = logs
                self.dispatch(Events.BATCH_END)
                self.state.reset(['logs'])
                
            self.dispatch(Events.EPOCH_END)
        self.dispatch(Events.TRAIN_END)
        self.dispatch(Events.RUN_END)
    
    def to_device(self,device,*args):
        return [arg.to(device) for arg in args]
                
    def test(self,model,test_dl):
       last_epoch = self.load_checkpoint(model)
       if last_epoch is None or last_epoch==0:
           self.logger.warning(f'Testing an untrained model !!!.')
       sample = next(iter(test_dl))
       audio,audio_frames,video = sample
    #    audio,audio_frames,video = self.to_device('cpu',audio,audio_frames,video)
       model.test(audio,audio_frames,video)

  
    