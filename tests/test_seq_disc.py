import torch


from wav2mov.params import params
from wav2mov.models import SequenceDiscriminator

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__file__)

def test():
    model = SequenceDiscriminator(params['disc']['sequence_disc'])
    frames = torch.randn(1,10,3,256,256)
    out = model(frames)
    logger.debug(f'out shape {out.shape}')
    assert(out.shape==(1,256))
def main():
    test()
    return
if __name__=='__main__':
    main()

