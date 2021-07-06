import torch
from typing import NamedTuple
from collections import namedtuple
from torch.utils.data import DataLoader,Dataset,random_split


class DataloadersPack:


    def get_dataloaders(self,dataset:Dataset,splits,batch_size,seed):
        
        train_set,test_set,validation_set = self.__split_dataset(dataset,splits,seed)
        train_dataloader = self.__get_dataloader(train_set,batch_size)
        test_dataloader = self.__get_dataloader(test_set,batch_size)
        validation_dataloader = self.__get_dataloader(validation_set,batch_size)
        
        dataloaders_pack = namedtuple('dataloaders_pack',['train','validation','test'])
        return dataloaders_pack(train_dataloader,validation_dataloader,test_dataloader)
    
    def __get_dataloader(self,dataset,batch_size):
    
        return DataLoader(dataset,
                          batch_size=batch_size,#,shuffle=True)
                          num_workers=2, #[WARNING] runs project.py this number of times
                          shuffle=True)
        
    def __split_dataset(self,dataset,splits,seed):
        N = len(dataset)
        train_size,test_size,val_size = self.__get_split_sizes(N,splits)
        return random_split(dataset,[train_size,test_size,val_size],
                            generator=torch.Generator().manual_seed(seed))
    
    def __get_split_sizes(self,N,splits):
        train_size = int((splits.train_size/10)*N)
        test_size = int((splits.test_size/10)*N)
        validation_size = N-train_size-test_size
        return train_size,test_size,validation_size
    
    


def splitted_dataloaders(dataset:Dataset,splits:namedtuple,batch_size,seed)->NamedTuple:
    
    return DataloadersPack().get_dataloaders(dataset,splits,batch_size,seed)