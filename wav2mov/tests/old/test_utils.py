import os
import torch
import unittest 

from test_settings import TEST_DIR

from wav2mov.utils.plots import save_gif


class TestUtils(unittest.TestCase):
    def test_save_gif_1(self):
        images = torch.randn(10,3,256,256)
        save_gif(os.path.join(TEST_DIR,'logs','test_1.gif'),images)
    def test_save_gif_2(self):
        images = []
        for _ in range(5):
            images.append(torch.randn(3,256,256))
        save_gif(os.path.join(TEST_DIR, 'logs', 'test_2.gif'), images)

def main():
    unittest.main()
    
if __name__=='__main__':
    main()
