import torch 
import logging
import unittest
from wav2mov.utils.audio import StridedAudioV2


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
class TestAudioUtil(unittest.TestCase):
    def test_frames_cont(self):
        BATCH_SIZE = 2
        audio = torch.randn(BATCH_SIZE,668*40)
    
        for coarticulation_factor in range(5):
            strided_audio = StridedAudioV2(666,coarticulation_factor)
            get_frames_from_idx,_ = strided_audio.get_frame_wrapper(audio)
            logging.debug('For coarticulation factor of {}'.format(coarticulation_factor))
            for i in range((audio.shape[0]//666)):
                frame = get_frames_from_idx(i)
                if i%10==0:
                    logging.debug(f'{i},{frame.shape}')
                self.assertEqual(frame.shape, (1,(coarticulation_factor*2 + 1)*666))
            
    def test_frames_range(self):
        BATCH_SIZE = 2
        audio = torch.randn(BATCH_SIZE,668*40)
    
        for coarticulation_factor in range(5):
            strided_audio = StridedAudioV2(666,coarticulation_factor)
            _,get_frames_from_range = strided_audio.get_frame_wrapper(audio)
            logging.debug('For coarticulation factor of {}'.format(coarticulation_factor))
            num_frames=5
            for i in range((audio.shape[0]//666)):
                frame = get_frames_from_range(i,i+num_frames-1)
                if i%10==0:
                    logging.debug(f'{i},{frame.shape}')
                self.assertEqual(frame.shape, (1,(num_frames+2)*666))
                
if __name__ == '__main__':
    unittest.main()
