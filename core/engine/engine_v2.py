class BaseEngine:
    def __init__(self):
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