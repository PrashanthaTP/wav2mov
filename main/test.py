import os
import time
from numpy import save


import torch
from torchvision import transforms as vtransforms
from torchvision import utils as vutils

from wav2mov.models.generator import GeneratorBW
from wav2mov.models.wav2mov_model import Wav2MovBW
from wav2mov.main.data import get_dataloaders
from wav2mov.utils.audio import StridedAudio
from wav2mov.utils.plots import show_img,save_gif
# from wav2mov.


# GEN_CHECKPOINT = r'E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\wav2mov\wav2mov\runs\Run_18_3_2021__2_4\gen_Run_18_3_2021__2_4.pt'



def test_model(options,hparams, config, logger):
    checkpoint = options.model_path
    loaders,mean,std= get_dataloaders(config, hparams, shuffle=True)
    val_dl = loaders.val
    stride = hparams['data']['audio_sf']//hparams['data']['video_fps']
    num_channels = hparams['img_channels']
    
    transforms = vtransforms.Compose(
        [vtransforms.Grayscale(1),
         vtransforms.Resize((hparams['img_size'], hparams['img_size'])),
         vtransforms.Normalize(mean,std)
        ]
    )

    strided_audio = StridedAudio(stride=stride, coarticulation_factor=0)
  
    device = torch.device('cpu')


    model = GeneratorBW(hparams=hparams['gen'])
   
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    

    sample = next(iter(val_dl))
    sample = next(iter(val_dl))

       

    audio, video = sample  


    video = video.permute(0, 1, 4, 2, 3)
    video=video/255  #! important
    audio, video = audio.to(device), video.to(device)

    get_framewise_audio = strided_audio.get_frame_wrapper(audio)
 
    still_image = video[:, -5, :, :, :]
    still_image = transforms(still_image)
    # vutils.save_image(still_image,'still_image.png')
    # return
    
    num_video_frames = video.shape[1]
    
    num_audio_frames = audio.shape[1]//stride
    
    limit = min(num_audio_frames, num_video_frames)
    
    fake_frames = []
    real_frames = []
    logger.info('started')
    for idx in range(limit):
        video_frame = video[:, idx, ...]  # ellipsis
        audio_frame, _ = get_framewise_audio(idx)
        video_frame = transforms(video_frame)
        # print(video_frame.shape)
        # show_img(video_frame)
        # return
        real_frames.append((video_frame*std +mean)*255)
        fake_frame = model(audio_frame, still_image)
        fake_frames.append((fake_frame.detach()*std +mean)*255)  
        
        #normalization involves (x-mean)/srd : -1 and 1
        
        logger.info(f'[{idx+1:2d}/{limit}] fake frame generated | shape {fake_frame.shape}')
    
    
    
    fake_frames = torch.cat(fake_frames,dim=0)
    real_frames = torch.cat(real_frames,dim=0)

    logger.info(f'fake_frames shape : {fake_frames.shape}')
    version = os.path.basename(options.model_path).strip('gen').split('.')[0]
    
    
    out_dir  = os.path.join(config['out_dir'],version)
    os.makedirs(out_dir,exist_ok=True)
    gif_name_fake = f'fake_frames_{version}.gif'
    gif_name_real = f'real_frames_{version}.gif'
    video_fps = hparams['data']['video_fps']
    save_gif(os.path.join(out_dir,gif_name_fake),fake_frames,duration=1/video_fps)
    save_gif(os.path.join(out_dir,gif_name_real),real_frames,duration=1/video_fps)
    
    vutils.save_image((still_image*std+mean)*255, os.path.join(out_dir,'still_frame.png'),normalize=True)
    vutils.save_image(fake_frames,
                      os.path.join(out_dir,f'test_fake_frames_{version}.png'),normalize=True)#if normalize option is not given output will be white boxes 
    vutils.save_image(real_frames,
                      os.path.join(out_dir,f'test_real_frames_{version}.png'),normalize=True)
