from typing import List 
import torch 



from wav2mov.core.engine.callbacks import CallbackStates, History,Callbacks,CallbackDispatcher
from wav2mov.core.utils.average_meter import AverageMeter
from wav2mov.core.utils.misc import get_tqdm_iterator

class BaseEngine:
   
    __req_fns__ = [ 'forward']
    def __init__(self,logger,device):
        self.logger = logger
        self.device = 'cpu'
        if device=='cuda': 
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
     
        
    def __is_fn_implemented(self,model,fn):
        fn_= getattr(model,fn,None)
        return callable(fn_)
    
 
    def safety_check(self,model):
        for fn in self.__req_fns__:
            if self.__is_fn_implemented(model,fn):
                continue
            raise(NotImplementedError(f'{fn} not implemented in {model.__class__.__name__} class.'))
     
    def run(self,*args,**kwargs):
        self.logger.info(f'Using {self.device}')


class Engine(BaseEngine):
    def __init__(self,config,logger):
        super().__init__(logger,device=config.get('device','cpu'))
        self.config = config 
        self.callbacks = []
        self._state : CallbackStates = None 

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state

    def dispatch(self, new_state, **kwargs):
        self.state = new_state
        if self._cb_dispatcher is not None:
            self._cb_dispatcher(new_state, **kwargs)
        else:
            raise Exception('Callback Dispatcher not set.')
    
    def run(self,model,callbacks_list:List[Callbacks]):
        """Runs training and validation loops.

        Args:
        `model` (nn.Module): Model to be trained
            callbacks_list (List[Callbacks]):
            can include
            + :py:class:`~.callbacks.TensorboardCallback`
            + custom callback inheriting from :py:class:`~callbacks.Callbacks`
        """
        super().run()
        if callbacks_list is None:
            callbacks_list = []
        self.history = History()
        callbacks_list.append(self.history)
        self._cb_dispatcher = CallbackDispatcher(callbacks_list, model)

        dataloaders = model.setup_dataloaders()
        train_dl, validation_dl = dataloaders.train,dataloaders.validation 
        self.total_epochs = model.hparams.num_epochs
        ##############################
        # Model Training
        ##############################
        self.logger.info('Training Started')
        model.to(device=self.device)
        self.dispatch(CallbackStates.TRAIN_START)
        logs = {}
        for epoch in range(1, self.total_epochs+1):
            self.dispatch(CallbackStates.EPOCH_START, epoch=epoch)
            logs['train']  = self.__train(model, train_dl,epoch)
            logs['validation'] = self.__validate(model, validation_dl,epoch)
            self.dispatch(CallbackStates.EPOCH_END, epoch=epoch,logs = logs)
            
        self.dispatch(CallbackStates.TRAIN_END, logs=logs)
        self.logger.info('Training Successfully Completed.')
        return self.history

    def __train(self,model,train_dl,epoch) ->dict:
        """Training steps per epoch

        Args:
            model (nn.Module): Model to be trained
            train_dl (DataLoader): Data Src
            epoch (int): Current Epoch
            total_epochs (int): Total Number of Epochs
        
        Returns : 
            dict : Contains average loss value of the epoch
        """
        model.train()
        # losses = AverageMeter(name="train_loss", fmt=":0.2f")
        self.dispatch(CallbackStates.TRAIN_EPOCH_START, epoch=epoch)
        # prog_bar = get_tqdm_iterator(train_dl, description=f'[TRAIN]({epoch}/{self.total_epochs})',colour='blue')

        train_logs = {}

        for batch_idx, batch in enumerate(train_dl):
            self.dispatch(CallbackStates.TRAIN_STEP_START, batch_idx=batch_idx,epoch=epoch)

            data, targets = batch
            data, targets = data.to(self.device),targets.to(self.device)
            train_logs = model.train_step(data, targets)
            # losses.update(train_logs['metrics']['loss'], n=data.size(0))
            # prog_bar_logs = {key:f"{value:0.2f}" for key,value in train_logs['metrics'] }
            # prog_bar.set_postfix(prog_bar_logs)

            self.dispatch(CallbackStates.TRAIN_STEP_END, batch_idx=batch_idx,epoch=epoch,logs=train_logs)


        self.dispatch(CallbackStates.TRAIN_EPOCH_END, epoch=epoch,logs=train_logs) #used by History and Tensorboard callbacks 
        return train_logs


    def __validate(self,model,validation_dl,epoch) ->dict:
        """Performs Validation steps per epoch

        Args:
            model (nn.Module): Model to be trained
            val_dl (DataLoader): Data Src
            epoch (int): Current Epoch
            total_epochs (int): Total Number of Epochs
            
        Returns : 
         dict : contains avg loss value of the epoch
        """
        model.eval()
    
        self.dispatch(CallbackStates.VAL_EPOCH_START, epoch=epoch)
        prog_bar = get_tqdm_iterator(validation_dl, description=f'[VALIDATION]{epoch}',colour='yellow')

        validation_logs = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_bar):

                data, targets = batch
                data, targets = data.to(self.device),targets.to(self.device)

                self.dispatch(CallbackStates.VAL_STEP_START, batch_idx=batch_idx,epoch=epoch)

                validation_logs = model.validation_step(data, targets)

                prog_bar_logs = {key:f"{value:0.2f}" for key,value in validation_logs['metrics'] }
                prog_bar.set_postfix(prog_bar_logs)
                
                self.dispatch(CallbackStates.VAL_STEP_END, batch_idx=batch_idx,epoch=epoch,logs=validation_logs)

 
        self.dispatch(CallbackStates.VAL_EPOCH_END, epoch=epoch,logs=validation_logs) #used by History and Tensorboard callbacks 
        return validation_logs

class GanEngine(BaseEngine):
    def __init__(self, config,logger):
        super().__init__(use_gpu=config.get('use_gpu',False))
        
    
    