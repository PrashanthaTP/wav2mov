""" Training script : can start from previous checkpoint"""
import os
import time


import torch
from torchvision import transforms as vtransforms

from wav2mov.models.wav2mov_2 import Wav2MovBW
from wav2mov.main.data import get_dataloaders
from wav2mov.utils.audio import StridedAudio
from wav2mov.utils.misc import AverageMetersList, ProgressMeter

from wav2mov.logger import TensorLogger


DISPLAY_EVERY = 5


def setup_tensor_logger(config):
    tensor_logger = TensorLogger(config['runs_dir'])
    writer_names = ['writer_gen', 'writer_sync_disc',
                    'writer_seq_disc', 'writer_id_disc']
    tensor_logger.add_writers(writer_names)
    return tensor_logger


def get_tensor_logger(options, config):
    return setup_tensor_logger(config) if options.log in ['y', 'yes'] else None


def add_to_board(tensor_logger, losses, global_step, scalar_type):
    for name, value in losses.items():
        writer_name = 'writer_' + name
        tensor_logger.add_scalar(
            writer_name, scalar_type+'_'+name, value, global_step)


def add_img_grid(tensor_logger, img_grid, global_step, img_type):
    tensor_logger.add_image(img_grid, img_type, global_step)


def get_train_dl(config, hparams):
    loaders, mean, std = get_dataloaders(config, hparams, shuffle=True)
    train_dl = loaders.train
    return train_dl, mean, std


def get_transforms(img_size, img_channels):
    transforms = vtransforms.Compose(
        [
            vtransforms.Grayscale(1),
         vtransforms.Resize(img_size),
         vtransforms.Normalize([0.5]*img_channels, [0.5]*img_channels)
         ])
    return transforms


def get_device(hparams):
    if hparams['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


def load_checkpoint(options):
    return torch.load(options.model_path)

    

def get_meters(hparams):
    num_epochs = hparams['num_epochs']
    loss_meters = AverageMetersList(('id_disc', 'sync_disc', 'seq_disc', 'gen'),
                                    fmt=':0.4f')  # per video
    epoch_loss_meters = AverageMetersList(('id_disc', 'sync_disc', 'seq_disc', 'gen'),
                                          fmt=':0.4f')  # per epoch

    progress_meter = ProgressMeter(num_epochs, epoch_loss_meters.as_list())

    return loss_meters, epoch_loss_meters, progress_meter


def train_model(options, hparams, config, logger):
    train_dl, mean, std = get_train_dl(config, hparams)

    img_channels = hparams['img_channels']
    img_size = hparams['img_size']
    num_epochs = hparams['num_epochs']

    transforms = get_transforms((img_size, img_size), img_channels)

    stride = hparams['data']['audio_sf']//hparams['data']['video_fps']
    strided_audio = StridedAudio(stride=stride, coarticulation_factor=0)
    device = get_device(hparams)

    logger.info(f'option : num_videos : {options.num_videos} | mean : {mean} std: {std} | stride :{stride}')

    start_epoch = 0
    model = Wav2MovBW(config, hparams, logger)
    if getattr(options,'model_path',None) is not None:
        logger.debug(f'Loading pretrained weights : {config.version}')
       
        prev_epoch = model.load(checkpoint_dir=options.model_path)
        if prev_epoch is not None:
            start_epoch = prev_epoch+1
        logger.debug(f'weights loaded successfully: {config.version}')
 
    NUM_VIDEOS = options.num_videos if options.num_videos is not None else len(train_dl)
    ################################
    # Setup loggers and loss meters
    ################################
    tensor_logger = get_tensor_logger(options, config)
    loss_meters, epoch_loss_meters, progress_meter = get_meters(hparams)

    STILL_IMAGE_IDX = 5
    logger.info(f'{STILL_IMAGE_IDX}th frame from last of every video is considered as reference image for the generator')
    logger.info(f'Training started on {device}')
    steps = 0

    start_time = time.time()
    model.on_train_start()

    ################################
    # Training loop
    ################################
    for epoch in range(start_epoch,num_epochs):
        epoch_start_time = time.time()
        epoch_loss_meters.reset()
        for batch_idx, sample in enumerate(train_dl):
            batch_start_time = time.time()
            loss_meters.reset()

            audio, video = sample  # channel axis is the last one

            # channel axis must be after the batch size,frame_count
            #change video shape from (B,F,H,W,C) to (B,F,C,H,W)
            video = video.permute(0, 1, 4, 2, 3)

            audio, video = audio.to(device), video.to(device)
            video = video/255  # !important
            get_framewise_audio = strided_audio.get_frame_wrapper(audio)
            # video is of shape(batch_size,num_video_frames,channels,H,W)

            # video is of shape : (batch_size,num_frames,channels,img_height,img_width)
            num_video_frames = video.shape[1]

            num_audio_frames = audio.shape[1]//stride
            limit = min(num_audio_frames, num_video_frames)

            still_image = video[:, -STILL_IMAGE_IDX, :, :, :]
            still_image = transforms(still_image)
            model.set_condition(still_image)
            
            for idx in range(limit):
                video_frame = video[:, idx, ...]  # ellipsis
                audio_frame, _ = get_framewise_audio(idx)
                video_frame = transforms(video_frame)

                model.set_input(audio_frame, video_frame)
                losses = model.optimize_parameters()

                loss_meters.update(losses, n=1)
                steps += 1

            losses = model.optimize_sequence()
            gen_loss = losses.pop('gen', 0)
            loss_meters.update(losses, n=1)
            loss_meters.get('gen').add(gen_loss)

            batch_duration = time.time()-batch_start_time
            # for every video as batch size is 1
            if tensor_logger is not None:
                add_to_board(tensor_logger, loss_meters.average(),
                             steps, scalar_type='loss')

            epoch_loss_meters.update(loss_meters.average(), n=1)

            logger.debug( f'\nEpoch {epoch+1}/{num_epochs} [{batch_duration:0.2f} s or {batch_duration/60:0.2f} min] : video num {batch_idx+1}/{len(train_dl)}')

            logger.debug(f'audio shape : {audio.shape} | video shape : {video.shape}')

            logger.debug(loss_meters)

            if (batch_idx+1) == NUM_VIDEOS:
                break
            if (batch_idx) % 20 == 0:
                model.save(epoch=epoch)
                hparams.save(config['params_checkpoint_fullpath'])

        logger.info(progress_meter.get_display_str(epoch+1))
        model.save(epoch=epoch)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info( f'[Epoch {epoch+1}/{num_epochs}] {epoch_duration:0.2f} seconds or {epoch_duration/60:0.2f} minutes')

    hparams.save(config['params_checkpoint_fullpath'])
    model.on_train_end()

    end_time = time.time()
    total_train_time = end_time - start_time
    logger.info('Trainging successfully completed')
    logger.info(f'Time taken {total_train_time:0.2f} seconds or {total_train_time/60:0.2f} minutes')
    logger.info(str(model.to('cpu')))
    
    model_path = config['gen_checkpoint_fullpath']
    with open('trained.txt', 'w') as file:
        file.write(model_path)
