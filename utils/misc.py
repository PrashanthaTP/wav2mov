import time

def log_run_time(logger):
    def timeit(func):
        def timed(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            end_time = time.time()
            time_taken = end_time-start_time
            logger.log(f'[TIME TAKEN] {func.__name__} took {time_taken:0.2f} seconds (or) {time_taken/60:0.2f} minutes',type="DEBUG")
            return res
        return timed
    return timeit



class AverageMeter:
    def __init__(self,name,fmt=':0.4f'):
      
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def _update_average(self):
        self.avg = self.sum/self.count 
        
    def update(self,val,n):
        self.count +=n
        self.sum += val*n 
        self._update_average()
        
    def __str__(self):
        fmt_str = '{name} : {avg' + self.fmt + '}'
        return fmt_str.format(**self.__dict__)
    
    def add(self,val):
        self.sum += val
        self._update_average()
    
class AverageMetersList:
    def __init__(self, names, fmt=':0.4f'):
        self.meters = {name:AverageMeter(name,fmt) for name in names}
        
    def update(self,d:dict):
        """update the average meters
        
        Args:
            d (dict): key is the name of the meter and value is a tuple containing value and the multiplier
            
        """
        for name,(value,n) in d.items():
            self.meters[name].update(value,n)

    
    def reset(self):
        for name in self.meters.keys():
            self.meters[name].reset()
    
    def as_list(self):
        return self.meters.values()
    
    def average(self):
        return {name:meter.avg for name,meter in self.meters.items()}
    
    def get(self,name):
        if name not in self.meters:
            raise KeyError(f'{name} has not average meter initialized')
        return self.meters.get(name)
    
    def __str__(self):
        avg = self.average()
        return '\t'.join(f'{key}:{val:0.4f}' for key,val in avg.items())
    
class ProgressMeter:
    def __init__(self,steps,meters,prefix=''):
        self.batch_fmt_str = self._get_epoch_fmt_str(steps)
        self.meters = meters
        self.prefix = prefix
    
    def get_display_str(self,step):
        entries = [self.prefix + self.batch_fmt_str.format(step)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)
    
    def _get_epoch_fmt_str(self,steps):
        num_digits = len(str(steps//1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt +'/'+ fmt.format(steps) + ']'



def get_duration_in_minutes_seconds(self,duration):
    return duration//60,duration