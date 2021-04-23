import torch
import logging

from wav2mov.config import get_config
from wav2mov.params import params

from wav2mov.models.wav2mov_v7 import Wav2MovBW


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
config = get_config('v_test')

# BATCH_SIZE = params['data']['batch_size']
BATCH_SIZE = 1 

NUM_FRAMES = 5

def get_input():
    audio = torch.randn(BATCH_SIZE,666*NUM_FRAMES,device='cuda')
    video = torch.randn(BATCH_SIZE,NUM_FRAMES,1,256,256,device='cuda')
    audio_frames = torch.randn(BATCH_SIZE,NUM_FRAMES,666+4*666,device='cuda')
    return audio,video,audio_frames
def test():
    model = Wav2MovBW(config,params,logger)
    audio,video,audio_frames = get_input()
    
    model.on_train_start()
    for epoch in range(1):
        model.on_batch_start(1)
        audio_frames = audio_frames.reshape(-1,audio_frames.shape[-1])
        frame_img = video[:,NUM_FRAMES//2,...]
        model.set_input(frame_img,audio,audio_frames,video) 
        model.optimize_parameters()
        model.optimize_sequence()
        

if __name__ == '__main__':
    test()