import torch

from wav2mov.core.data.collates import get_batch_collate
from wav2mov.params import params
from wav2mov.config import get_config
from wav2mov.logger import get_module_level_logger

from wav2mov.main.data import get_dataloaders_v2
from wav2mov.main.options import Options

logger = get_module_level_logger(__name__)
def test(options,hparams,config):
    collate_fn = get_batch_collate(hparams['data'])
    dataloader_pack = get_dataloaders_v2(options,config,params,collate_fn=collate_fn)
    train_dl,test_dl = dataloader_pack
    logger.debug(f'train : {len(train_dl)} test : {len(test_dl)}')
    
    dl_iter = iter(train_dl)
    for _ in range(min(len(train_dl),10)):
        sample = next(dl_iter)
        audio,video = sample.audio,sample.video
        logger.debug(f'video {video.shape} : {torch.mean(video,dim=[0,1,3,4])} ,{torch.std(video,dim=[0,1,3,4])}')
        logger.debug(f'audio {audio.shape} : {torch.mean(audio)} ,{torch.std(audio)}')
    
def main():
    options = Options().parse()
    config = get_config(options.version)
    test(options,params,config)
    
if __name__ == '__main__':
    main()
     

