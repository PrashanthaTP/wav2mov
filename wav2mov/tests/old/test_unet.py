import torch
import unittest
# from wav2mov.models.generator import GeneratorBW
from wav2mov.models.generator_v6 import GeneratorBW,Generator,Encoder,Decoder

params = {
    "img_dim":[256,256],
    "retain_dim":True,
    "device":"cpu",
    "in_channels": 1,
		"enc_chs": [64, 128, 256, 512, 1024],
		"dec_chs": [1024, 512, 256, 128, 64],
		"up_chs": [1026, 512, 256, 128, 64],
  }



class TestUnet(unittest.TestCase):
    def test_encoder(self):
        encoder = Encoder(chs=[params['in_channels']] +params['enc_chs'])
        image = torch.randn(1,1,256,256)
        out = encoder(image)
        print('testing encoder ')
        for layer in out:
            print(layer.shape)
        
    def test_decoder(self):
        encoder = Encoder(chs=[params['in_channels']] + params['enc_chs'])
        image = torch.randn(1,1,256,256)
        encoded = encoder(image)[::-1]
        
        audio_noise_encoded = torch.randn(1,2,8,8)
        encoded[0] = torch.cat([encoded[0],audio_noise_encoded],dim=1)
        decoder = Decoder(up_chs=params['up_chs'],dec_chs=params['dec_chs'])
        image = torch.randn(1,1,256,256)
        out = decoder(encoded[0],encoded[1:])
        print('test_decoder : ',out.shape)
        
    def test_gen(self):
        gen = Generator(params)
        frame_img = torch.randn(1,1,256,256)
        audio_noise = torch.randn(1,2,8,8)
        out = gen(frame_img,audio_noise)
        self.assertEqual(out.shape,(1,1,256,256))
        
def test():
    with torch.no_grad():
        x = torch.randn(1,1,256,256)
        a = torch.randn(1,666)
        gen = GeneratorBW( {
                        "device": "cpu",
                        "in_channels": 1,
                        "enc_chs": [64, 128, 256, 512, 1024],
                        "dec_chs": [1024, 512, 256, 128, 64],
                        "up_chs": [1026, 512, 256, 128, 64],
                        "img_dim": [256, 256],
                        "retain_dim": True,
                            "lr": 1e-4
                            })   
        gen.eval()
        assert gen(a,x).shape==(1,1,256,256)
        print("Test Passed " ,1)
      
if __name__ == '__main__':
    unittest.main()
