import os
import re
import torch
from torchvision import utils as vutils
from wav2mov.core.data.collates import get_batch_collate
from wav2mov.models.wav2mov_inferencer import Wav2movInferencer
from wav2mov.main.data import get_dataloaders_v2 as get_dataloaders

class Evaluator:
    def __init__(self,config):
        self.config = config
    
    def process_sample(self,sample):
        pass
    def run(self,model,test_dl):
        pass      

def squeeze_frames(video):
    batch_size,num_frames,_ = video.shape 
    return video.reshape(batch_size*num_frames,_) 

def make_path_compatible(path):
    if os.sep != '\\':#not windows
        return re.sub(r'(\\)+',os.sep,path)
    
def test_model(options,hparams,config,logger):
    logger.debug(f'Testing model...')
    version = os.path.basename(options.model_path).strip('gen_').split('.')[0]
    out_dir = config['out_dir']
    model = Wav2movInferencer(hparams)
    checkpoint = torch.load(options.model_path)
    state_dict,last_epoch = checkpoint['state_dict'],checkpoint['epoch']
    model.load(state_dict)
    logger.log(f'model was trained for {last_epoch+1} epochs. ')  
    collate_fn = get_batch_collate(hparams['data']) 
    test_dl = get_dataloaders(options,config,hparams,collate_fn=collate_fn)
    sample = next(iter(test_dl))

    audio,audio_frames,video = sample
    fake_video_frames = model.test(audio_frames,video)
    fake_video_frames = squeeze_frames(fake_video_frames) 
    video = squeeze_frames(video)
    os.makedirs(out_dir,exist_ok=True)
    save_path_fake_video_frames = os.path.join(out_dir,f'fake_frames_{version}.png')
    save_path_real_video_frames = os.path.join(out_dir,f'real_frames_{version}.png')
    save_path_fake_video_frames = make_path_compatible(save_path_fake_video_frames)
    save_path_real_video_frames = make_path_compatible(save_path_real_video_frames)
    
    vutils.save_image(fake_video_frames,save_path_fake_video_frames)
    vutils.save_image(video,save_path_real_video_frames)
    logger.debug(f'results are saved in {out_dir}')
    
    