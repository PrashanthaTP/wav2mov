
from enum import Enum
class CallbackEvents(Enum):
    TRAIN_START = "on_train_start"
    TRAIN_END = "on_train_end"
    EPOCH_START = "on_epoch_start"
    EPOCH_END = "on_epoch_end"
    BATCH_START = "on_batch_start"
    BATCH_END = "on_batch_end"
    RUN_START = "on_run_start"
    RUN_END = "on_run_end"
    NONE = "none"

class Callbacks:
    def on_train_start(self, *args,**kwargs):
         pass
    def on_train_end(self, *args,**kwargs): 
        pass
    def on_epoch_start(self, *args,**kwargs): 
        pass
    def on_epoch_end(self,*args, **kwargs): 
        pass
    def on_batch_start(self,*args, **kwargs): 
        pass
    def on_batch_end(self, *args,**kwargs): 
        pass
    def on_run_start(self,*args,**kwargs):
        pass
    def on_run_end(self,*args,**kwargs):
        pass
    
class CallbackDispatcher:
    def __init__(self):
        self.callbacks = []
    
    def register(self,callbacks):
        self.callbacks = callbacks
        
    def dispatch(self,event,*args,**kwargs):
        for callback in self.callbacks:
            # print(callback.__class__.__name__,args)
            getattr(callback,event.value)(*args,**kwargs)
            
