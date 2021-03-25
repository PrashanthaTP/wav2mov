from enum import Enum
from typing import List

import logging 

from torch.utils.tensorboard import SummaryWriter

from wav2mov.core.utils.misc import get_tqdm_iterator



class Callbacks:
    def on_train_start(self, **kwargs): pass
    def on_train_end(self, **kwargs): pass
    def on_epoch_start(self, **kwargs): pass
    def on_epoch_end(self, **kwargs): pass
    def on_train_step_start(self, **kwargs): pass
    def on_train_step_end(self, **kwargs): pass
    def on_val_step_start(self, **kwargs): pass
    def on_val_step_end(self, **kwargs): pass
    def on_train_epoch_start(self, **kwargs): pass
    def on_train_epoch_end(self, **kwargs): pass
    def on_val_epoch_start(self, **kwargs): pass
    def on_val_epoch_end(self, **kwargs): pass


class CallbackStates(Enum):
    TRAIN_START = "on_train_start"
    TRAIN_END = "on_train_end"
    EPOCH_START = "on_epoch_start"
    EPOCH_END = "on_epoch_end"
    TRAIN_STEP_START = "on_train_step_start"
    TRAIN_STEP_END = "on_train_step_end"
    VAL_STEP_START = "on_val_step_start"
    VAL_STEP_END = "on_val_step_end"
    TRAIN_EPOCH_START = "on_train_epoch_start"
    TRAIN_EPOCH_END = "on_train_epoch_end"
    VAL_EPOCH_START = "on_val_epoch_start"
    VAL_EPOCH_END = "on_val_epoch_end"


def CallbackHook():
    state: CallbackStates = None

    def set_state(new_state):
        state = new_state

    def get_state():
        return state

    return set_state, get_state


class CallbackDispatcher:
    """Class for running callbacks
        [Ref] : tez library by Abhishek K R Thakur 
        
        [link] https://github.com/abhishekkrthakur/tez
    """

    def __init__(self, callbacks_list: List[Callbacks], model):
        self.model = model
        self.callbacks = callbacks_list

    def __call__(self, state, **kwargs):
        for cb in self.callbacks:
            #.value since model_state is an enum
            getattr(cb, state.value)(model=self.model, **kwargs)

    


class History(Callbacks):
    """Callback attached to each run of engine 
        [Ref] : Tensorflow/Keras Callbacks 
        
        [link] : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras
    """

    def __init__(self):
        self.history = {}
        self.history['train'] = None
        self.history['validation'] = None

    def on_train_epoch_start(self, model, epoch):
        if self.history['train'] is None:
            self.history['train'] = {}

    def on_val_epoch_start(self, model, epoch):
        if self.history['validation'] is None:
            self.history['validation'] = {}

    def on_train_epoch_end(self, model, epoch, logs=None):
        history_logs = logs.get('metrics',None) if logs else None 
        for key,val in history_logs.items():
            self.history['train'].setdefault(key, []).append(val)

    def on_val_epoch_end(self, model, epoch, logs=None):
        history_logs = logs.get('metrics',None) if logs else None 
        for key,val in history_logs.items():
            self.history['validation'].setdefault(key, []).append(val)


class TqdmBar:
    def __init__(self):
        self.bar = None

    @classmethod
    def from_tqdm(cls, iterable, description, colour='green'):
        prog_bar = cls()
        prog_bar.bar = get_tqdm_iterator(iterable, description, colour=colour)
        return prog_bar

    def update(self,d:dict):
        if self.bar is None:
            raise AttributeError(f'{self.__class__.__name__} was not initialized properly')
        self.bar.set_postfix(d)

    def clear(self):
        self.bar.set_postfix_str('')
        
class Progbar(Callbacks):
    def __init__(self):
        pass 
    def configure_train_bar():
        pass 
    def on_train_step_end(self, model, batch_idx, epoch, logs=None):
        # progbar_logs = logs.get_type('prog_bar') if logs is not None else None 
        progbar_logs = logs.get('progbar',None)  if logs else None 
        if progbar_logs:
            self.update(progbar_logs)

    def on_val_step_end(self, model, batch_idx, epoch, logs=None):
        progbar_logs = logs.get('progbar', None) if logs else None
        if progbar_logs:
            self.update(progbar_logs)

class TensorBoardTargets(Enum):
    IMG_GRID = "img_grid" 
    SCALAR = "scalar"
    
class TensorboardCallback(Callbacks):
    def __init__(self, logdir,targets:dict):
        self.writer = SummaryWriter(logdir)
        self.targets = targets 
        
    def on_train_epoch_end(self, model, epoch, logs=None):
        board_logs = logs.get('tensorboard',None) if logs else None
        if board_logs is None : 
            logging.warning('Tensorboard Callback configured,but no values were passed on train epoch end.')
            return 
        if not all(name in board_logs for name in self.target.values()):
            raise ValueError('[ERROR] [TensorboardCallback] Key not present for tensorboard logging.Please make sure to pass all the values as passed during initialization of callback')
        for target_type,name in self.targets:
            if target_type == TensorBoardTargets.IMG_GRID:
                self.writer.add_image(name,board_logs[name],global_step=epoch) 
            elif target_type == TensorBoardTargets.SCALAR:
                self.writer.add_scalar(f'train/{name}',board_logs[name], global_step=epoch)

    def on_val_epoch_end(self, model, epoch, logs=None):
        board_logs = logs.get('tensorboard',None) if logs else None
        if board_logs is None : 
            logging.warning('Tensorboard Callback configured,but no values were passed on validation epoch end.')
            return 
        if not all(name in board_logs for name in self.target.values()):
            raise ValueError('[ERROR] [TensorboardCallback] Key not present for tensorboard logging.Please make sure to pass all the values as passed during initialization of callback')
        for target_type,name in self.targets:
            if target_type == TensorBoardTargets.IMG_GRID:
                self.writer.add_image(name,board_logs[name],global_step=epoch) 
            elif target_type == TensorBoardTargets.SCALAR:
                self.writer.add_scalar(f'validation/{name}',board_logs[name], global_step=epoch)

# Currently only compares with loss : which is kind of hardcoded
# TODO : Make it dynamic with an option to save the model based on whether it is required to maximize/minimize a metric.


class ModelCheckpoint(Callbacks):
    """Callback for saving models
    """

    def __init__(self, checkpoint_location):
        self.min_loss = float('inf')
        self.checkpoint_location = checkpoint_location

    def on_val_epoch_end(self, model, epoch, logs:None):
        if logs is None :
            logging.warning('ModelCheckpoint Callback configured,but no metric/loss values were passed on validation epoch end.')
            return 
        curr_loss = logs['metrics']['loss']
        if curr_loss < self.min_loss:
            self.min_loss = curr_loss
            model.save_checkpoint(model, self.min_loss,self.checkpoint_location, save_params=False)

    def on_train_end(self, model, logs):
        loss = None
        if 'validation' in logs and 'metrics' in logs['validation'] and 'loss' in logs['metrics']['validation']:
            loss = logs['validation']['metrics']['loss']
        else:
            loss = logs['train']['metrics']['loss']
        model.save_checkpoint(model, loss, self.checkpoint_location, save_params=True)
