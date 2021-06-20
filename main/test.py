import os
import re
import torch
from torchvision import utils as vutils

from scipy.io.wavfile import write

from wav2mov.core.data.collates import get_batch_collate
from wav2mov.models.wav2mov_inferencer import Wav2movInferencer
from wav2mov.main.data import get_dataloaders
from wav2mov.utils.plots import save_gif,save_video

SAMPLE_NUM = 10



def squeeze_frames(video):
    batch_size,num_frames,*extra = video.shape 
    return video.reshape(batch_size*num_frames,*extra) 

def denormalize_frames(frames):
  return ((frames*0.5)+0.5)*255
  
def make_path_compatible(path):
    if os.sep != '\\':#not windows
        return re.sub(r'(\\)+',os.sep,path)
    
def test_model(options,hparams,config,logger):
    logger.debug(f'Testing model...')
    version = os.path.basename(options.model_path).strip('gen_').split('.')[0]
    logger.debug(f'loading version : {version}')
    out_dir = os.path.join(config['out_dir'],version)
    model = Wav2movInferencer(hparams,logger)
    checkpoint = torch.load(options.model_path,map_location='cpu')
    state_dict,last_epoch = checkpoint['state_dict'],checkpoint['epoch']
    model.load(state_dict)
    logger.debug(f'model was trained for {last_epoch+1} epochs. ')  
    collate_fn = get_batch_collate(hparams['data']) 
    logger.debug('Loading dataloaders')
    dataloaders = get_dataloaders(options,config,hparams,collate_fn=collate_fn)
    test_dl = dataloaders.val
    sample = next(iter(test_dl))
    for _ in range(SAMPLE_NUM-1):
      sample = next(iter(test_dl))
    # print(sample,len(sample))
    audio,audio_frames,video = sample
    fake_video_frames,ref_video_frame = model.test(audio_frames,video)
    fake_video_frames = squeeze_frames(fake_video_frames) 
    video = squeeze_frames(video)
    os.makedirs(out_dir,exist_ok=True)
    save_path_fake_video_frames = os.path.join(out_dir,f'fake_frames_{version}.png')
    save_path_real_video_frames = os.path.join(out_dir,f'real_frames_{version}.png')
    save_path_ref_video_frame = os.path.join(out_dir,f'ref_frame_{version}.png')
    save_path_fake_video_frames = make_path_compatible(save_path_fake_video_frames)
    save_path_real_video_frames = make_path_compatible(save_path_real_video_frames)
    save_path_ref_video_frame = make_path_compatible(save_path_ref_video_frame)
    
    
    save_video(hparams['data'],os.path.join(out_dir,'fake_video_without_audio.avi'),audio,denormalize_frames(fake_video_frames))
    logger.debug(f'video saved : {fake_video_frames.shape}')
    
    vutils.save_image(denormalize_frames(ref_video_frame),save_path_ref_video_frame,normalize=True)
    vutils.save_image(denormalize_frames(fake_video_frames),save_path_fake_video_frames,normalize=True)
    vutils.save_image(denormalize_frames(video),save_path_real_video_frames,normalize=True)

    gif_path = os.path.join(out_dir,f'fake_frames_{version}.gif')
    save_gif(gif_path,denormalize_frames(fake_video_frames))
    gif_path = os.path.join(out_dir,f'real_frames_{version}.gif')
    save_gif(gif_path,denormalize_frames(video))
    write(os.path.join(out_dir,f'audio_{SAMPLE_NUM}.wav'),16000,audio.cpu().numpy().reshape(-1))
    logger.debug(f'audio saved : audio_{SAMPLE_NUM}.wav')

    logger.debug(f'results are saved in {out_dir}')
    
    msg = f'test_run for version {version}.\n'
    msg += '='*25
    msg += f'\nlast epoch : {last_epoch}\n'
    msg += f'curr_version : {config.version}\n'
    with open(os.path.join(out_dir,'info.txt'),'a+') as file:
      file.write(msg)
      
    