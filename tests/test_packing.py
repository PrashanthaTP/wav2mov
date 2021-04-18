from collections import namedtuple
import torch
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence

from wav2mov.core.data.collates import collate_fn

from wav2mov.main.data import get_dataloaders
from wav2mov.params import params
from wav2mov.config import config

import argparse
def parse_args():
    arg_parser = argparse.ArgumentParser(description='Testing Variable Sequence Batch Size')
    arg_parser.add_argument('--samples','-s',type=int,default=1,help='number of samples whose details will be printed')
    arg_parser.add_argument('--batch_size','-b',type=int,default=10,help='batch size for dataloader')
    return arg_parser.parse_args()




    
def test_batch_size(options):

    dl,_,_ = get_dataloaders(config,params,get_mean_std=False,collate_fn=collate_fn)
    dl = dl.train
    for i in range(options.samples):
        sample,lens = next(iter(dl))
        audios,videos = sample
        audio_lens,video_lens = lens
        print(audios.shape)
        videos = pack_padded_sequence(videos,lengths=video_lens,batch_first=True,enforce_sorted=False)
        audios = pack_padded_sequence(audios, lengths = audio_lens,batch_first=True,enforce_sorted = False)
        print(videos,audios)
        print(f' sample {i+1} '.center(30,'='))
        # print('video shape : ',sample.video.shape)
        # print('audio shape : ',sample.audio.shape)
     
        print('video shape : ',videos.data.shape)
        print('audio shape : ',audios.data.shape)
        print('batch sizes : ',videos.batch_sizes)
        # print('lens audio : ',lens.audio)
        # print('lens video :',lens.video)
        p ,l= pad_packed_sequence(videos,batch_first=True)
        print(p.shape,l)
        print(''*30)
        
    
if __name__ == '__main__':
    options = parse_args()
    BATCH_SIZE = options.batch_size
    data = params['data']
    params.set('data', {**data, 'batch_size': BATCH_SIZE})
    test_batch_size(options)
