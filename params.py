import json 
import os 
from wav2mov.settings import BASE_DIR
from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)
class Params:
    def __init__(self):
        self.vals = {}
       
    

    def _flatten_dict(self, d):
        flattened = {}
        for k, v in d.items():
            if isinstance(v, dict):
                inner_d = self._flatten_dict(v)
                flattened = {**flattened, **inner_d}
            else:
                flattened[k] = v
        return flattened

    def update(self,key,value):
        if key in self.vals:
            logger.warning(f'Updating existing parameter {key} : changing from {self.vals[key]} with {value}')
        
        self.vals[key] = value
        
    def _update_vals_from_dict(self, d: dict):
        self.vals = d
        # d = self._flatten_dict(d)
        # self.vals = {**self.vals,**d}

    @classmethod
    def from_json(cls, json_file_fullpath):
        with open(json_file_fullpath, 'r') as file:
            configs = json.load(file)
            obj = cls()
            obj._update_vals_from_dict(configs)
            return obj

    def __getitem__(self, item):
        if item not in self.vals:
            logger.error(f'{self.__class__.__name__} object has no key called {item}')
            raise KeyError(f'No key called {item}')
       
        return self.vals[item]

    def __repr__(self):
        fstr = ''
        for key,val in self.vals.items():
            fstr += f"{key} : {val}\n"
        return fstr
    
    def set(self,key,val):
        if key in self.vals:
            logger.warning(f'updating existing value of {key} with value {self.vals[key] } to {val}')
        self.vals[key] = val
        
    def save(self,file_fullpath):
        with open(file_fullpath,'w') as file:
            json.dump(self.vals,file)


#Singleton Object
params = Params.from_json(os.path.join(BASE_DIR, 'params.json'))

if __name__ == '__main__':
    pass
