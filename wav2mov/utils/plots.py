import os
import torch
from torch.functional import Tensor
from torchvision.io import write_video

import imageio
import numpy as np
from matplotlib import pyplot as plt

from moviepy import editor as mpy
from scipy.io.wavfile import write as write_audio

import warnings
warnings.filterwarnings( "ignore", module = r"matplotlib\..*" )

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)

def no_grad_wrapper(fn):
    def wrapper(*args,**kwargs):
        with torch.no_grad():
            return fn(*args,**kwargs)
    return wrapper

@no_grad_wrapper
def show_img(img,cmap='viridis'):
    if isinstance(img,np.ndarray):
        img_np = img
    else:
        if len(img.shape)>3:
            img = img.squeeze(0)
        img_np = img.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    # print(img_np.shape)
    # print(img_np)
    if img_np.shape[2]==1:#if single channel
        img_np = img_np.squeeze(2)
    plt.imshow(img_np,cmap=cmap)
    plt.show()
    # 

def save_gif(gif_path,images,duration=0.5):    
    """creates gif 

    Args:
        gif_path (str): path where gif file should be saved
        images (torch.funcitonal.Tensor): tensor of images of shape (N,C,H,W)
    """
    if isinstance(images,Tensor):
        images = images.numpy()
  

    images = images.transpose(0,2,3,1).astype('uint8')
   
    imageio.mimsave(gif_path,images,duration=0.5)
    
def save_video(hparams,video_path,audio,video_frames):
    """
      audio_frames : C,S
      video_frames : T,C,H,W
    """
    if video_frames.shape[1]==1:
      video_frames = video_frames.repeat(1,3,1,1)
    logger.debug(f'video frames :{video_frames.shape}, audio : {audio.shape}')
    video_frames = video_frames.to(torch.uint8)
    write_video(filename= video_path,
            video_array = video_frames.permute(0,2,3,1),
            fps = hparams['video_fps'],
            video_codec="h264",
            # audio_array= audio,
            # audio_fps = hparams['audio_sf'],
            # audio_codec = 'mp3'
          )
    dir_name = os.path.dirname(video_path)
    temp_audio_path = os.path.join(dir_name,'temp','temp_audio.wav')
    os.makedirs(os.path.dirname(temp_audio_path),exist_ok=True)
    write_audio(temp_audio_path,hparams['audio_sf'],audio.cpu().numpy().reshape(-1))

    video_clip = mpy.VideoFileClip(video_path)
    audio_clip = mpy.AudioFileClip(temp_audio_path)
    video_clip.audio = audio_clip
    video_clip.write_videofile(os.path.join(dir_name,'fake_video_with_audio.avi'),fps=hparams['video_fps'],codec='png')
    
def save_video_v2(hparams,filepath,audio,video_frames):
    def get_video_frames(idx):
        idx = int(idx)
        # logger.debug(f'{video_frames.shape} ,{video_frames[idx].shape}')
        frame = video_frames[idx].permute(1,2,0).squeeze()
        return frame.cpu().numpy().astype(np.uint8)

    logger.debug('saving video please wait...')
    num_frames = video_frames.shape[0]
    video_fps = hparams['data']['video_fps']
    audio_sf = hparams['data']['audio_sf']
    duration = audio.squeeze().shape[0]/audio_sf
    # duration = 10
    # duration = math.ceil(num_frames/video_fps)
    logger.debug(f'duation {duration} seconds')
    dir_name = os.path.dirname(filepath)
    temp_audio_path = os.path.join(dir_name,'temp','temp_audio.wav')
    os.makedirs(os.path.dirname(temp_audio_path),exist_ok=True)
    # print(audio.cpu().numpy().reshape(-1).shape)
    write_audio(temp_audio_path,audio_sf,audio.cpu().numpy().reshape(-1))
    
    video_clip = mpy.VideoClip(make_frame=get_video_frames,duration=duration)
    audio_clip = mpy.AudioFileClip(temp_audio_path,fps=audio_sf)
    video_clip = video_clip.set_audio(audio_clip)
    # print(filepath,video_clip.write_videofile.__doc__)
    video_clip.write_videofile( filepath,
                                fps=video_fps,
                                codec="png",
                                bitrate=None,
                                audio=True,
                                audio_fps=audio_sf,
                                preset="medium",
                                # audio_nbytes=4,
                                audio_codec=None,
                                audio_bitrate=None,
                                # audio_bufsize=2000,
                                temp_audiofile=None,
                                # temp_audiofile_path="",
                                remove_temp=True,
                                write_logfile=False,
                                threads=None,
                                ffmpeg_params=['-s','256x256','-aspect','1:1'],
                                logger="bar",
                                # pixel_format='gray
                                )
