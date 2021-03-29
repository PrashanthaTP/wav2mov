import torch
from wav2mov.models.generator import GeneratorBW

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
    test()
