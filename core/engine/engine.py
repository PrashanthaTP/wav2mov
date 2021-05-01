import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.DEBUG)
from .callbacks import CallbackEvents,CallbackDispatcher


class TemplateEngine:

    def __init__(self):
        self.__event = CallbackEvents.NONE
        self.__dispatcher = CallbackDispatcher()     
    
    @property
    def event(self):
        return self.__event
    
    @event.setter
    def event(self,event):
        self.__event = event
        
    def register(self,callbacks):
        self.__dispatcher.register(callbacks)
        
    def dispatch(self,event,*args,**kwargs):
        self.event = event
        # logger.debug(f'{self.event},args : {args} kwargs : {kwargs}')
        self.__dispatcher.dispatch(event,*args,**kwargs)
        
    def on_run_start(self,*args,**kwargs):
        pass
    def on_run_end(self,*args,**kwargs):
        pass
    def on_train_start(self,*args,**kwargs):
        pass
    def on_epoch_start(self,*args,**kwargs):
        pass
    def on_batch_start(self,*args,**kwargs):
        pass
    def on_batch_end(self,*args,**kwargs):
        pass
    def log(self,*args,**kwargs):
        pass
    def validate(self,*args,**kwargs):
        pass
    def on_epoch_end(self,*args,**kwargs):
        pass 
    def on_train_end(self,*args,**kwargs):
        pass
    def run(self,*args,**kwargs):
        """
        training script goes here  
        """
        print("[TEMPLATE ENGINE] 'run' function not implemented")
        pass