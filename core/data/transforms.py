"""
Module for different data transforms
classes
----------------------
+ ToTensor
+ TargetNormalize
+ TargetStandardize
+ StandardScalar

"""
from torch import from_numpy
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
      
        data, target = sample['data'], sample['target']

        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        # # Matplot lib expects image in W x H x C
        # image = image.transpose((2, 0, 1))
        return {'data': from_numpy(data).float(),
                'target': from_numpy(target).float()}
        


class TargetNormalize(object):
    """Normalizes the target"""
   
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
        
    def __call__(self,sample):
        target = sample['target']
        target = (target-self.mean)/self.std
        return {'data':sample['data'],'target':target}


class TargetStandardize(object):
    def __init__(self,max_val,min_val):
        self.max_val = max_val
        self.min_val = min_val
        
    def __call__(self,sample):
        target = sample['target']
        target = (target-self.min_val)/(self.max_val-self.min_val)
        return {'data':sample['data'],'target':target}
    

class StandardScalar(object):
    def fit(self,data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        
    def transform(self,data):
        data -= self.mean
        data /= (self.std+ 1e-7)
        return data
    
    def fit_transform(self,data):
        mean = np.mean(data)
        std = np.std(data)
        data -= mean
        data /= (std+ 1e-7)
        return data