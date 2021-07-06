import torch
from wav2mov.params import params
from wav2mov.config import get_config
from wav2mov.models.generator.audio_encoder import AudioEnocoder

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__file__)
config = get_config('test_sync')
def test():
    model = AudioEnocoder(params['gen'])
    audio = torch.randn(2,10,7,13)
    out = model(audio)
    logger.debug(f'out shape {out.shape}')
    assert(out.shape==(2,10,params['gen']['latent_dim_audio']))
    
def main():
    test()
    
if __name__=='__main__':
    main()
