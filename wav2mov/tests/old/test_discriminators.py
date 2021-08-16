import torch
import unittest

from wav2mov.models.identity_discriminator import IdentityDiscriminator
from wav2mov.models.sequence_discriminator import SequenceDiscriminator
from wav2mov.models.sync_discriminator import SyncDiscriminator
from wav2mov.models.patch_disc import PatchDiscriminator

from utils import get_input_of_shape,no_grad
BATCH_SIZE = 1
IMAGE_SIZE = 256
CHANNELS = 3




class TestDescs(unittest.TestCase):
    @no_grad
    def test_identity(self):
        x = get_input_of_shape((BATCH_SIZE,CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
        y = get_input_of_shape((BATCH_SIZE,CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
        desc = IdentityDiscriminator()
        out = desc(x,y)
        print(f"identity descriminator : input :{x.shape} and {y.shape} | output : {out.shape}")
        self.assertEqual(out.shape,(BATCH_SIZE,16))
        
    @no_grad
    def test_sequence(self):
        image_size = IMAGE_SIZE*IMAGE_SIZE*3
        hidden_size = 100
      
        desc = SequenceDiscriminator(image_size,hidden_size,num_layers=1)
        x = get_input_of_shape((BATCH_SIZE,2,image_size))
        out = desc(x)
        print(f"sequence descriminator : input :{x.shape}  | output : {out.shape}")
        self.assertEqual(out.shape,(BATCH_SIZE,hidden_size))
        
    @no_grad
    def test_sync(self):
        audio = get_input_of_shape((BATCH_SIZE,666))
        image = get_input_of_shape((BATCH_SIZE,CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
        desc = SyncDiscriminator()
        out = desc(audio,image)
        print(f"sync descriminator : input :{audio.shape} and {image.shape} | output : {out.shape}")
        self.assertEqual(out.shape,(BATCH_SIZE,128))
        
    @no_grad
    def test_patch_disc(self):
        frame_image = get_input_of_shape((BATCH_SIZE,1,256,256))
        still_image = get_input_of_shape((BATCH_SIZE,1,256,256))
        disc = PatchDiscriminator(1,ndf=64)
        out = disc(frame_image,still_image)
        print(f'patch disc out: {out.shape}')
        self.assertEqual(out.shape,(BATCH_SIZE,1,30,30))
        
def main():
    unittest.main()
    return

if __name__=='__main__':
    main()