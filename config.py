"""Config"""
import json
import os
import logging 
from datetime import datetime

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_curr_run_str():
    now = datetime.now()
    date,time = now.date(),now.time()
    day,month,year = date.day,date.month,date.year
    hour , minutes = time.hour,time.minute
    return 'Run_{}_{}_{}__{}_{}'.format(day,month,year,hour,minutes)

class Config :
    def __init__(self):
        self.vals = {'base_dir':BASE_DIR}
        self.version = get_curr_run_str()
        #these two are used to decide whether to create a path using os.makedirs
        self.fixed_paths = set()
        self.runtime_paths = set()
     
    def _rectify_paths(self,val):
        return val % {'base_dir': self.vals['base_dir'],
                      'version': self.version,
                      'checkpoint_filename': 'checkpoint_'+self.version,
                      'log_filename': 'log_' + self.version
                      }
    
    def _flatten_dict(self,d):
        flattened= {}
        for k,v in d.items():
            if isinstance(v,dict):
                inner_d = self._flatten_dict(v)
                for k_inner in inner_d.keys():
                    if k== 'fixed': self.fixed_paths.add(k_inner)
                    else: self.runtime_paths.add(k_inner)
                    
                flattened = {**flattened,**inner_d}
            else:
                flattened[k] = v
        return flattened
                
    def _update_vals_from_dict(self,d:dict):
        
        d = self._flatten_dict(d)
        for key,val in d.items():
            if isinstance(val,str):
                value = self._rectify_paths(val)
                
            else:
                value = val

            # setattr(self,key,val)
            self.vals[key] = value
            
    @classmethod
    def from_json(cls,json_file_fullpath):
        with open(json_file_fullpath,'r') as file:
            configs = json.load(file)
            obj = cls()
            obj._update_vals_from_dict(configs)
            return obj
    
    def __getitem__(self,item):
        if item not in self.vals:
            logger.error(f'{self.__class__.__name__}  object has no key called {item}')
            raise KeyError('No key called ',item)
        if item in self.fixed_paths:
            return self.vals[item]
        if 'fullpath' in item or 'dir' in item:
            path = os.path.dirname(self.vals[item]) if '.' in os.path.basename(self.vals[item]) else self.vals[item]
            os.makedirs(path,exist_ok=True)
            logger.debug(f'directory created/accessed : {path}')
        return self.vals[item]

    
config = Config.from_json(os.path.join(BASE_DIR,'config.json'))
if __name__=='__main__':
    pass
