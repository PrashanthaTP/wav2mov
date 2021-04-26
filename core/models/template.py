from torch import nn 

class TemplateModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,*args,**kwargs):
        raise NotImplementedError("Forward method must be implemented")
    
    def on_train_start(self,*args,**kwargs):
        pass
    def on_epoch_start(self,*args,**kwargs):
        pass
    def on_batch_start(self,*args,**kwargs):
        pass
    def setup_input(self,*args,**kwargs):
        pass
    def optimize_parameters(self,*args,**kwargs):
        pass
    def on_batch_end(self,*args,**kwargs):
        pass
    def on_epoch_end(self,*args,**kwargs):
        pass
    def log(self,*args,**kwargs):
        pass
    def validate(self,*args,**kwargs):
        pass
    def on_train_end(self,*args,**kwargs):
        pass