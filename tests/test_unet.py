import torch
from wav2mov.models.generator import Generator 

def test():
    with torch.no_grad():
        x = torch.randn(1,3,572,572)
        gen = Generator()
        gen.eval()
        assert gen(x).shape==(1,3,256,256)
        print("Test Passed " ,1)
      
if __name__ == '__main__':
    test()