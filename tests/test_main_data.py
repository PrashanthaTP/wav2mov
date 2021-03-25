import unittest

from wav2mov.main.data import get_dataloaders
from wav2mov.config import config 
from wav2mov.params import params

class TestData(unittest.TestCase):
    def test_dataloader(self):
        dataloader,mean,std = get_dataloaders(config,params,shuffle=True)
        
        print(f'train dl : {len(dataloader.train)}')
        print(f'val dl : {len(dataloader.val)}')
        print(f'channel wise mean : {mean}')
        print(f'channel wise std : {std}')

if __name__ == '__main__':
    unittest.main()