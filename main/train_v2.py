from wav2mov.models.wav2mov import Wav2Mov 
from wav2mov.main.engine import Engine
from wav2mov.main.callbacks import LossMetersCallback,ModelCheckpoint,TimeTrackerCallback

from wav2mov.core.data.collates import get_batch_collate
from wav2mov.main.data import get_dataloaders



def train_model(options,hparams,config,logger):
    engine = Engine(options,hparams,config,logger)
    model = Wav2Mov(hparams,config,logger)
    collate_fn = get_batch_collate(hparams['data'])
    dataloaders_ntuple = get_dataloaders(options,config,hparams,
                                         get_mean_std=False,
                                         collate_fn=collate_fn)
    callbacks = [LossMetersCallback(options,hparams,logger,
                                    verbose=True),
                 TimeTrackerCallback(hparams,logger),
                 ModelCheckpoint(model,hparams,config,
                                 save_every=5)]
    
    engine.run(model,dataloaders_ntuple,callbacks)