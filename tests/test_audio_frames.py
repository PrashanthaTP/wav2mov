import numpy
import os
import torch
from matplotlib import pyplot as plt
from scipy.io.wavfile import write

from wav2mov.config import get_config
from wav2mov.params import params as hparams

from wav2mov.main.options import Options
from wav2mov.core.data.collates import get_batch_collate
from wav2mov.main.data import get_dataloaders as get_dl

DIR = os.path.dirname(os.path.abspath(__file__))
def test():
    options = Options().parse()
    config = get_config('test_audio_frames')
    collate_fn = get_batch_collate(hparams['data'])
    dls = get_dl(options,config,hparams,collate_fn=collate_fn)
    train_dl = dls.train
    sample = next(iter(train_dl))
    audio,audio_frames,video = sample
    
    print(f'audio : {audio.shape}')
    print(f'audio frames : {audio_frames.shape}')
    print(f'video : {video.shape}')
    for audio_frame in audio_frames[0][:10]:
        print(f'max mfccs {torch.max(audio_frame)}')
        print(f'min mfccs {torch.min(audio_frame)}')
        print(f'mean mfccs {torch.mean(audio_frame)}')
        print(f'std mfccs {torch.std(audio_frame)}')
    plt.imshow(audio_frames[0][0])
    plt.show()
    
if __name__ == '__main__':
    test()
    
    