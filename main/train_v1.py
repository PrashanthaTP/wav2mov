import os
import time


import torch
from torchvision import transforms as vtransforms

from wav2mov.models.wav2mov_v1 import Wav2MovBW
from wav2mov.main.data import get_dataloaders
from wav2mov.utils.audio import StridedAudio
from wav2mov.utils.misc import AverageMetersList,ProgressMeter
# from wav2mov.
from wav2mov.logger import TensorLogger


DISPLAY_EVERY = 5
def setup_tensor_logger(config):
    tensor_logger = TensorLogger(config['runs_dir'])
    writer_names = ['writer_gen', 'writer_sync_disc',
                    'writer_seq_disc', 'writer_id_disc']
    tensor_logger.add_writers(writer_names)
    return tensor_logger

def add_to_board(tensor_logger, losses, global_step,scalar_type):
    for name, value in losses.items():
        writer_name = 'writer_' + name
        tensor_logger.add_scalar(writer_name, scalar_type+'_'+name, value, global_step)
        
def add_img_grid(tensor_logger,img_grid,global_step,img_type):
    tensor_logger.add_image(img_grid,img_type,global_step)


def train_model(options,hparams, config, logger):
    loaders, mean,std = get_dataloaders(config, hparams, shuffle=True)
    logger.info(f'option : num_videos : {options.num_videos} | mean : {mean} std: {std}')
    train_dl = loaders.train
    stride = hparams['data']['audio_sf']//hparams['data']['video_fps']
    transforms = vtransforms.Compose(
        [vtransforms.Grayscale(1), 
         vtransforms.Resize((hparams['img_size'],hparams['img_size'])),
         vtransforms.Normalize(mean,std)
        ])


    strided_audio = StridedAudio(stride=stride, coarticulation_factor=0)
    if hparams['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Wav2MovBW(config, hparams, logger)
   
    num_epochs = hparams['num_epochs']

    if options.log in ['y','yes']:
        tensor_logger = setup_tensor_logger(config)
    else:
        tensor_logger = None
    loss_meters = AverageMetersList(('id_disc', 'sync_disc', 'seq_disc', 'gen'),
                                    fmt=':0.4f')#per video
    epoch_loss_meters = AverageMetersList(('id_disc', 'sync_disc', 'seq_disc', 'gen'),
                                          fmt=':0.4f')#per epoch
    
    progress_meter = ProgressMeter(num_epochs,epoch_loss_meters.as_list())

    STILL_IMAGE_IDX = 5
    logger.info(f'{STILL_IMAGE_IDX}th frame from last of every video is considered as reference image for the generator and also last frame of every video is omitted.')
    logger.info(f'Training started on {device}')
    steps = 0
    
    start_time = time.time()
    model.on_train_start()  
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss_meters.reset()
        for batch_idx, sample in enumerate(train_dl):
            batch_start_time = time.time()
            loss_meters.reset()

            audio, video = sample #channel axis is the last one 

            video = video.permute(0,1,4,2,3) #channel axis must be after the batch size,frame_count
        
            audio, video = audio.to(device), video.to(device)
            video = video/255#!important
            get_framewise_audio = strided_audio.get_frame_wrapper(audio)
            # video is of shape(batch_size,num_video_frames,channels,H,W)
            #? Think  again : which frame should be considered as reference image?
            still_image = video[:, -STILL_IMAGE_IDX, :, :, :]# considering 10 the frame from last hoping the speaker closes his/her mouth and is in rest postion with respect to talking
            still_image = transforms(still_image)
            model.set_condition(still_image)
       
            num_video_frames = video.shape[1] #video is of shape : (batch_size,num_frames,channels,img_height,img_width)
            
            """
            #? What happens if the batch size is not 1
            * cuda could file a case against you for not giving enough space in order to do your job.
            ! worst effect that could take place if batch size is not `1`
            ! Laptop being going to a mood of not responding to your queries
            """
            
            
            num_audio_frames = audio.shape[1]//stride 
            limit = min(num_audio_frames,num_video_frames)
            for idx in range(limit):
                video_frame = video[:,idx,...]#ellipsis 
                audio_frame,_ = get_framewise_audio(idx)
                video_frame = transforms(video_frame)
           
            
                model.set_input(audio_frame, video_frame)
                losses = model.optimize_parameters()

                loss_meters.update(losses, n=1)
                steps += 1
                
            batch_duration = time.time()-batch_start_time
            # for every video as batch size is 1
            if tensor_logger is not None:
                add_to_board(tensor_logger, loss_meters.average(),  steps,scalar_type='loss')
                
            epoch_loss_meters.update(loss_meters.average(),n=1)
            logger.debug(f'\nEpoch {epoch+1}/{num_epochs} [{batch_duration:0.2f} s or {batch_duration/60:0.2f} min] : video num {batch_idx+1}/{len(train_dl)}')
            logger.debug(f'audio shape : {audio.shape} | video shape : {video.shape}')
            logger.debug(loss_meters.average())
            if (batch_idx+1)==options.num_videos:
                break
            if (batch_idx)%20==0:
                model.save()
                hparams.save(config['params_checkpoint_fullpath'])
                
            
        logger.info(progress_meter.get_display_str(epoch))
        model.save()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f'[Epoch {epoch+1}/{num_epochs}] {epoch_duration:0.2f} seconds or {epoch_duration/60:0.2f} minutes')

    hparams.save(config['params_checkpoint_fullpath'])
    model.on_train_end()
    
    end_time = time.time()
    total_train_time = end_time - start_time
    logger.info('Trainging successfully completed')
    logger.info(f'Time taken {total_train_time:0.2f} seconds or {total_train_time/60:0.2f} minutes')
    logger.info(str(model.to('cpu')))