
class AverageMeter():
    """class for tracking metrics generated during training
    """
    def __init__(self,name,fmt=':0.2f'):
        self.name = name
        self.fmt = fmt 
        self.reset()
        
    def reset(self):
        self.curr_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self,val,n=1):
        self.curr_val = val 
        self.sum += val*n 
        self.count += n
        self.avg = self.sum / self.count 
        