import logging
import os
import torch
from torchvision.io import write_video
from scipy.io.wavfile import write as write_audio
from moviepy import editor as mpy

from inference import params
def get_module_level_logger(name):
    m_logger = logging.getLogger(name)
    m_logger.setLevel(logging.DEBUG)
    m_logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-5s : %(filename)s :%(lineno)s | %(asctime)s | %(msg)s ", "%b %d,%Y %H:%M:%S"))
    m_logger.addHandler(handler)
    return m_logger

logger = get_module_level_logger('utils')

def save_video(video_path, audio, video_frames):
    """
      audio_frames : C,S
      video_frames : F,C,H,W
    """
    # if has batch dimension remove it
    hparams = {'video_fps':params.VIDEO_FPS,'audio_sf':params.AUDIO_SF}
    if len(video_frames.shape) == 5:
        video_frames = video_frames[0]
    if video_frames.shape[1] == 1:
        video_frames = video_frames.repeat(1, 3, 1, 1)
        logger.warning('Grayscale images...')
    logger.debug(f'✅ video frames :{video_frames.shape[:]}, audio : {audio.shape[:]}')
    video_frames = video_frames.to(torch.uint8)
    os.makedirs(os.path.dirname(video_path),exist_ok=True)
    write_video(filename=video_path,
                video_array=video_frames.permute(0, 2, 3, 1),
                fps=hparams['video_fps'],
                video_codec="h264",
                )
    dir_name = os.path.dirname(video_path)
    temp_audio_path = os.path.join(dir_name, 'temp', 'temp_audio.wav')
    os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)
    write_audio(temp_audio_path,hparams['audio_sf'], audio.cpu().numpy().reshape(-1))

    video_clip = mpy.VideoFileClip(video_path)
    audio_clip = mpy.AudioFileClip(temp_audio_path)
    video_clip.audio = audio_clip
    video_clip.write_videofile(os.path.join(dir_name, 'fake_video_with_audio.avi'), fps=hparams['video_fps'], codec='png',logger=None)
