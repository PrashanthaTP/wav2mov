from wav2mov.models.wav2mov_trainer import Wav2Mov 
from wav2mov.main.engine import Engine
from wav2mov.main.callbacks import LossMetersCallback,ModelCheckpoint,TimeTrackerCallback,LoggingCallback

from wav2mov.core.data.collates import get_batch_collate
from wav2mov.main.data import get_dataloaders



def train_model(options,hparams,config,logger):
    engine = Engine(options,hparams,config,logger)
    model = Wav2Mov(hparams,config,logger)
    collate_fn = get_batch_collate(hparams['data'])
    dataloaders_ntuple = get_dataloaders(options,config,hparams,
                                         collate_fn=collate_fn)
    callbacks = [LossMetersCallback(options,hparams,logger,
                                    verbose=True),
                LoggingCallback(options,hparams,config,logger),
                 TimeTrackerCallback(hparams,logger),
                 ModelCheckpoint(model,hparams,config,
                                 save_every=5)]
    
    engine.run(model,dataloaders_ntuple,callbacks)