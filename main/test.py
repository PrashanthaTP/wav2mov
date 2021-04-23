
import os
import re
import time
from numpy import save


import torch
from torchvision import transforms as vtransforms
from torchvision import utils as vutils

from wav2mov.models.generator import GeneratorBW
from wav2mov.models.wav2mov import Wav2MovBW
from wav2mov.main.data import get_dataloaders
from wav2mov.utils.audio import StridedAudioV2
from wav2mov.utils.plots import show_img, save_gif
# from wav2mov.
from wav2mov.main.utils import get_transforms

# GEN_CHECKPOINT = r'E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\wav2mov\wav2mov\runs\Run_18_3_2021__2_4\gen_Run_18_3_2021__2_4.pt'

def process_audio(audio, hparams):
    stride = hparams['data']['audio_sf']//hparams['data']['video_fps']
    # total len : (666+666)+666+(666+666)
    strided_audio = StridedAudioV2(stride=stride,
                                   coarticulation_factor=hparams['data']['coarticulation_factor'],
                                   device=hparams['device'])
    get_frames_from_idx, get_frames_from_range = strided_audio.get_frame_wrapper(
        audio)
    return get_frames_from_idx, get_frames_from_range


def process_video(video, hparams):

    img_channels = hparams['img_channels']
    img_size = hparams['img_size']
    transforms = get_transforms((img_size, img_size), img_channels)
    # channel axis must be after the batch size,frame_count
    #change video shape from (B,F,H,W,C) to (B,F,C,H,W)
    video = video.permute(0, 1, 4, 2, 3)

    video = video/255  # !important
    bsize, frames, channels, height, width = video.shape

    video = video.reshape(bsize*frames, channels, height, width)
    #if not done the `Resize transform` gets angry that it got 3d images
    # while it was assured previously that it will get 2d images
    video = transforms(video)
    video = video.reshape(
        bsize, frames, video.shape[-3], video.shape[-2], video.shape[-1])
    return video


def process_sample(sample, hparams):

    stride = hparams['data']['audio_sf']//hparams['data']['video_fps']
    device = hparams['device']
    audio, video = sample

    audio, video = audio.to(device), video.to(device)

    get_frames_from_idx, get_frames_from_range = process_audio(audio, hparams)
    video = process_video(video, hparams)

    num_video_frames = video.shape[1]
    num_audio_frames = audio.shape[1]//stride
    limit = min(num_audio_frames, num_video_frames)

    return get_frames_from_idx, get_frames_from_range, video, limit

def test_model(options, hparams, config, logger):

    checkpoint = options.model_path
    loaders, mean, std = get_dataloaders(
        options, config, hparams, shuffle=True)
    val_dl = loaders.val
    for _ in range(25):
        sample = next(iter(val_dl))
        sample = next(iter(val_dl))

    # sample = next(iter(loaders.val))


  

    model = GeneratorBW(hparams=hparams['gen'])
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    last_epoch = None
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        last_epoch = checkpoint['epoch']
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    get_frames_from_idx, get_frames_from_range, video, limit = process_sample(sample,hparams)
    still_image = video[:, -25, :, :, :]


    fake_frames = []
    real_frames = []
    logger.info('started')
    for idx in range(limit):
        video_frame = video[:, idx, ...]  # ellipsis
        audio_frame = get_frames_from_idx(idx)
    
        real_frames.append((video_frame*std + mean)*255)
        fake_frame = model(audio_frame, still_image)
        fake_frames.append((fake_frame.detach()*std + mean)*255)

        #normalization involves (x-mean)/srd : -1 and 1

        logger.info(f'[{idx+1:2d}/{limit}] fake frame generated | shape {fake_frame.shape}')

    fake_frames = torch.cat(fake_frames, dim=0)
    real_frames = torch.cat(real_frames, dim=0)

    logger.info(f'fake_frames shape : {fake_frames.shape}')
    version = os.path.basename(options.model_path).strip('gen_').split('.')[0]

    out_dir = os.path.join(config['out_dir'], version)
    os.makedirs(out_dir, exist_ok=True)

    logger.debug(f'GIFs are saved in {out_dir}')
    vutils.save_image((still_image*std+mean)*255,
                      os.path.join(out_dir, 'still_frame.png'), normalize=True)
    vutils.save_image(fake_frames,
                      os.path.join(out_dir, f'test_fake_frames_{version}.png'), normalize=True)  # if normalize option is not given output will be white boxes
    vutils.save_image(real_frames,
                      os.path.join(out_dir, f'test_real_frames_{version}.png'), normalize=True)
    
    readme_file = os.path.join(out_dir,'readme.txt')
    if os.sep != '\\':
        readme_file = re.sub(r'(\\)+',os.sep,readme_file)
    if last_epoch is not None:
        with open(readme_file,'a+') as file:
            file.write(f'Last epoch : {last_epoch}')
        