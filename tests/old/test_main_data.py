import unittest

from wav2mov.main.data import get_dataloaders
from wav2mov.config import config 
from wav2mov.params import params

import argparse
# argparser = argparse.ArgumentParser(description='test dataloader')
# argparser.add_argument('--num_videos','-n',type=int,help='number of videos')
# options = argparser.parse_args()
class Options:pass
options = Options()
options.num_videos=36
class TestData(unittest.TestCase):
    def test_dataloader(self):
        dataloader,mean,std = get_dataloaders(options,config,params,shuffle=True)
        sample = next(iter(dataloader.train))
        print(f'train dl : {len(dataloader.train)}')
        print(f'val dl : {len(dataloader.val)}')
        print(f'channel wise mean : {mean}')
        print(f'channel wise std : {std}')
        print(f'audio shape : {sample.audio.shape }')
        print(f'video shape : {sample.video.shape }')

if __name__ == '__main__':
    unittest.main()