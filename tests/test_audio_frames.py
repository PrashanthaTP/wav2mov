import numpy
import os
import torch
from scipy.io.wavfile import write

from wav2mov.config import get_config
from wav2mov.params import params as hparams

from wav2mov.main.options import Options
from wav2mov.core.data.collates import get_batch_collate
from wav2mov.main.data import get_dataloaders_v2 as get_dl

DIR = os.path.dirname(os.path.abspath(__file__))
def test():
    options = Options().parse()
    config = get_config('test_audio_frames')
    collate_fn = get_batch_collate(hparams['data'])
    dls = get_dl(options,config,hparams,collate_fn=collate_fn)
    train_dl = dls.train
    sample = next(iter(train_dl))
    audio,audio_frames,video = sample
    
    audio_dir = os.path.join(DIR,'res')
    os.makedirs(audio_dir,exist_ok=True)
    a = audio[0].numpy()
    print('audio ',audio.shape,a.shape)
    write(os.path.join(audio_dir,'test_audio.wav'),16000,a)
    
    f = audio_frames[0][15].numpy()
    print('frame ',audio_frames.shape,f.shape)
    write(os.path.join(audio_dir,'test_audio_frame.wav'),16000,f)
    
    print('saved audio ',audio_dir)
if __name__ == '__main__':
    test()
    
    