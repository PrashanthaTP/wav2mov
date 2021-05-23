import os
import torch
from torchvision import utils as vutils

def squeeze_frames(video):
    batch_size,num_frames,*extra = video.shape 
    return video.reshape(batch_size*num_frames,*extra) 

def denormalize_frames(frames):
  return ((frames*0.5)+0.5)*255
  
def make_path_compatible(path):
    if os.sep != '\\':#not windows
        return re.sub(r'(\\)+',os.sep,path)


def save_series(frames,config,i=0):
    frames = squeeze_frames(frames)
    save_dir = os.path.join(config['runs_dir'],'images')
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir,f'sync_frames_fake_{i}.png')
    vutils.save_image(denormalize_frames(frames),save_path,normalize=True)