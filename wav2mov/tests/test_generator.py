import numpy as np
import torch
import unittest

from wav2mov.models.generator.id_encoder import IdEncoder
from wav2mov.models.generator.id_decoder import IdDecoder
from wav2mov.models.generator.frame_generator import Generator
from wav2mov.logger import get_module_level_logger
from wav2mov.utils.plots import show_img
logger = get_module_level_logger(__name__)

class TestGen(unittest.TestCase):
    def test_id_encoder(self):
        logger.debug(f'id_encoder test')
        hparams = {
                    'in_channels':1,
                   'chs':[64,128,256,512,1024],
                   'latent_dim_id':(8,8)
                   } 
        
        id_encoder = IdEncoder(hparams)
        images = torch.randn(1,1,256,256)
        encoded,intermediates = id_encoder(images)
        req_h,req_w = 128,128
        for i,intermediate in enumerate(intermediates):
            self.assertEqual((req_h,req_w),intermediate.shape[-2:])
            logger.debug(f'{i} : {intermediate.shape}')
            req_h,req_w = req_h//2,req_w//2
        logger.debug(f'final encoded {encoded.shape}') 
        
    def test_id_decoder(self):
        logger.debug(f'id_decoder test')
        hparams = {
                'in_channels':1,
                'chs':[64,128,256,512,1024],
                'latent_dim':16,
                'latent_dim_id':(8,8)
                    }
        id_enocder = IdEncoder(hparams)
        id_decoder = IdDecoder(hparams) 
        images = torch.randn(1,1,256,256)
        encoded,intermediates= id_enocder(images)
        decoded = id_decoder(encoded,intermediates)
        self.assertEqual(decoded.shape,(1,1,256,256))

    def test_generator(self):
        logger.debug(f'test generator')
        
        hparams = {
                'in_channels':3,
                'chs':[64,128,256,512,1024],
                'latent_dim':16+256+10,
                'latent_dim_id':(8,8),
                'latent_dim_audio':256,
                'latent_dim_noise':10,
                'device':'cpu',
                'lr':2e-4
            }
               
        gen = Generator(hparams)
        ref_frames = torch.randn(1,5,3,256,256)
        audio_frames = torch.zeros(1,5,5*666)
        out = gen(audio_frames,ref_frames)
        self.assertEqual(out.shape,(1,5,3,256,256))
        
        # show_img(ref_frames[0][0],cmap='gray')
        # show_img(out[0][0],cmap='gray')
        
        test_on_real_img(gen)
        
def test_on_real_img(gen):
        gen.eval()
        ref_frames = torch.from_numpy(np.load(r'E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\wav2mov\wav2mov\datasets\grid_dataset_256_256\s10_l_bbat9p\video_frames.npy'))
        show_img(ref_frames[0].permute(2,0,1))
        ref_frames = ref_frames[0].permute(2,0,1).unsqueeze(0).unsqueeze(0).float()
        # logger.debug(ref_frames.shape)
        audio_frames = torch.randn(1,1,5*666)
        out = gen(audio_frames,ref_frames)
        show_img(out[0][0])
        
if __name__ == '__main__':
    unittest.main()