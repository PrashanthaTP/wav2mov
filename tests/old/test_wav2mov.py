import torch
import logging

from wav2mov.core.data.collates import get_batch_collate

from wav2mov.config import get_config
from wav2mov.params import params

from wav2mov.models.wav2mov import Wav2Mov
from wav2mov.main.engine import Engine
from wav2mov.main.callbacks import LossMetersCallback,TimeTrackerCallback,ModelCheckpoint
from wav2mov.main.data import get_dataloaders

from wav2mov.main.options import Options,set_options

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# BATCH_SIZE = params['data']['batch_size']
BATCH_SIZE = 1 

NUM_FRAMES = 25

def get_input():
    audio = torch.randn(BATCH_SIZE,666*9,device='cuda')
    video = torch.randn(BATCH_SIZE,NUM_FRAMES,1,256,256,device='cuda')
    audio_frames = torch.randn(BATCH_SIZE,NUM_FRAMES,666+4*666,device='cuda')
    return audio,video,audio_frames
        
def test(options,hparams,config,logger):
    engine = Engine(options,hparams,config,logger)
    model = Wav2Mov(hparams,config,logger)
    collate_fn = get_batch_collate(hparams['data'])
    dataloaders_ntuple = get_dataloaders(options,config,hparams,
                                         get_mean_std=False,
                                         collate_fn=collate_fn)
    callbacks = [LossMetersCallback(options,hparams,logger,
                                    verbose=True),
                 TimeTrackerCallback(hparams,logger),
                 ModelCheckpoint(model,hparams,
                                 save_every=5)]
    
    engine.run(model,dataloaders_ntuple,callbacks)
    
if __name__ == '__main__':
    options = Options().parse()
    set_options(options,params)
    config = get_config(options.version)
    test(options,params,config,logger)