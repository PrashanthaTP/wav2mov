


        
class Engine():
    def __init__(self,logger):
        self.logger = logger
        
    def configure(self,hparams,options,config):
        self.hparams = hparams
        self.options = options
        self.config = config
    
    def on_run_start(self,model,train_dl):
        #log 
        accumulation_steps = self.hparams['data']['batch_size']//self.hparams['data']['mini_batch_size']
        num_batches = len(train_dl)//accumulation_steps
        self.logger.info('options num_videos : {self.options.num_videos}')
        self.logger.info(f'train_dl : len(train_dl) :{len(train_dl)} : num_batches: {num_batches}')
    
    def resume_checkpoint(self,model):
        if getattr(self.options, 'model_path', None) is not None:
            loading_version = os.path.basename(self.options.model_path)
            self.logger.debug(f'Loading pretrained weights : {self.config.version} <== {loading_version}')

            prev_epoch = model.load(checkpoint_dir=self.options.model_path)
            if prev_epoch is not None:
                start_epoch = prev_epoch+1
            self.logger.debug(f'weights loaded successfully: {self.config.version} <== {loading_version}')
            
    def run(self,model,dataloaders):
        train_dl = dataloaders.train
        num_epochs = self.hparams['num_epochs']
        

    