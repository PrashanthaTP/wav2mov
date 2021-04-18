import torch 
import unittest
from wav2mov.params import params
from wav2mov.config import get_config

from wav2mov.core.data.utils import AudioUtil
from wav2mov.core.data.collates import get_batch_collate

from wav2mov.main.data import get_dataloaders

STRIDE = 666
audio_util = AudioUtil(2,stride=STRIDE)

class Options:
    num_videos = 9
    v = 'v_test'
options = Options()
config = get_config(options.v)

class TestBatching(unittest.TestCase):
    def test_audio_batching(self):
        AUDIO_LEN = 44200
        audio = torch.randn(AUDIO_LEN) 
        frames = audio_util.get_audio_frames(audio.unsqueeze(0))
        self.assertEqual(frames.shape,(AUDIO_LEN//STRIDE,STRIDE*(4+1)))
        
        
    def test_collate(self):
        params.update('data',{**params['data'],'mini_batch_size':2})
        params.update('data',{**params['data'],'coarticulation_factor':0})
        collate = get_batch_collate(hparams=params['data'])
        
        dl = get_dataloaders(options,config,params,get_mean_std=False,collate_fn=collate)
        
        batch_size = params['data']['mini_batch_size']
        # print(vars(dl.train))
        # self.assertEqual(len(dl.train),batch_size)
        
        for i,sample in enumerate(dl.train):
            audio = sample.audio
            audio_frames = sample.audio_frames
            video = sample.video
            print(f'Batch {i}')
            print(f'audio :{audio.shape}')
            print(f'audio frames :{audio_frames.shape}')
            print(f'video : {video.shape} ')
        
            self.assertEqual(sample.audio.shape[0],batch_size)
            
if __name__ == '__main__':
    unittest.main()