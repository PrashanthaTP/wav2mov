import torch
from wav2mov.params import params
from wav2mov.config import get_config
from wav2mov.models import SyncDiscriminator

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__file__)
config = get_config('test_sync')
def test():
    model = SyncDiscriminator(params['disc']['sync_disc'],config)
    audio = torch.randn(2,666*9)
    frames = torch.randn(2,5,3,256,256)
    out = model(audio,frames)[0]
    logger.debug(f'out shape {out.shape}')
    assert(out.shape==(2,1))
    
def main():
    test()
    
if __name__=='__main__':
    main()
