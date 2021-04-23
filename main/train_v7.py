""" Training script : can start from previous checkpoint"""

import os
import time

from wav2mov.core.engine.engine import BaseEngine
from wav2mov.models.wav2mov_v7 import Wav2MovBW
from wav2mov.utils.audio import StridedAudio, StridedAudioV2

from wav2mov.main.utils import (
    get_meters_v2 as get_meters,
    get_device,
    get_transforms,
    get_train_dl_v2 as get_train_dl,
    add_to_board,
    get_tensor_logger)

DISPLAY_EVERY = 5
STILL_IMAGE_IDX = 5


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
    video = video.reshape(bsize, frames, video.shape[-3], video.shape[-2], video.shape[-1])
    return video


def process_sample(sample, hparams):

    device = get_device(hparams)
    audio,audio_frames,video = sample
    audio = audio.to(device)
    video = video.to(device)
    audio_frames = audio_frames.to(device)
    video = process_video(video,hparams)
    return audio,audio_frames,video
def get_still_images(video):
    batch_size,num_frames,channels,height,width = video.shape 
    still_images= video[:, -STILL_IMAGE_IDX, :, :, :]
    still_images = still_images.repeat((1,num_frames,1,1,1))
    still_images = still_images.reshape((batch_size*num_frames,channels,height,width))
    return still_images
    
def train_model(options, hparams, config, logger):
    train_dl, mean, std = get_train_dl(options, config, hparams)

    num_epochs = hparams['num_epochs']
    accumulation_steps = hparams['data']['batch_size']//hparams['data']['mini_batch_size']

    num_batches = len(train_dl)//accumulation_steps
    logger.info(f'options numm_videos : {options.num_videos},mean :{mean},std :{std}')
    logger.info( f'train_dl : len(train_dl) :{len(train_dl)} : num_batches: {num_batches}')

    start_epoch = 0
    model = Wav2MovBW(config, hparams, logger)

    if getattr(options, 'model_path', None) is not None:
        loading_version = os.path.basename(options.model_path)
        logger.debug(
            f'Loading pretrained weights : {config.version} <== {loading_version}')

        prev_epoch = model.load(checkpoint_dir=options.model_path)
        if prev_epoch is not None:
            start_epoch = prev_epoch+1
        logger.debug(f'weights loaded successfully: {config.version} <== {loading_version}')

    ################################
    # Setup loggers and loss meters
    ################################
    tensor_logger = get_tensor_logger(options, config)
    batch_loss_meters, epoch_loss_meters, epoch_progress_meter, batch_progress_meter = get_meters(hparams, num_batches=num_batches)

    logger.info(f'{STILL_IMAGE_IDX}th frame from last of every video is considered as reference image for the generator')
    logger.info(f'Training started on {hparams["device"]} with batch_size {hparams["data"]["batch_size"]} ')
    steps = 0

    start_time = time.time()
    ################################
    # Training loop
    ################################
    model.on_train_start()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss_meters.reset()
        batch_loss_meters.reset()
        batch_duration = 0.0

        epoch_start_time = time.time()
        model.on_epoch_start(epoch)
        for batch_idx, sample in enumerate(train_dl):
            batch_start_time = time.time()
            audio,audio_frames,video = process_sample(sample,hparams) 
            still_images = get_still_images(video)
            
            model.on_batch_start(batch_idx)  # reset frames history
            
            model.set_input(still_images,audio,audio_frames,video)
            losses = model.optimize_parameters()#optimize id disc
            # losses of num_frames * mini_batch
            batch_loss_meters.update(losses, n=video.shape[0]*video.shape[1])

            losses = model.optimize_sequence()#optimize sync and sequence

            batch_duration += time.time() - batch_start_time

            # losses of sync and sequence
            batch_loss_meters.update(losses, n=video.shape[0])
            
            steps += 1

            if (batch_idx+1) % accumulation_steps == 0:
                model.batch_descent()

                # for every video as batch size is 1
                if tensor_logger is not None:
                    add_to_board(tensor_logger, batch_loss_meters.average(),
                                 steps, scalar_type='loss')

                epoch_loss_meters.update(
                    batch_loss_meters.average(), n=hparams['data']['batch_size'])

                logger.debug(batch_progress_meter.get_display_str((batch_idx+1)//hparams['data']['batch_size']))
                logger.debug(f"Batch duration : {batch_duration:0.4f} seconds or {batch_duration/60:0.4f} minutes")

                batch_loss_meters.reset()
                batch_duration = 0.0

            if (batch_idx+1) % 5 == 0:
                model.save(epoch=epoch)
                logger.debug(
                    f'model saved in {config["gen_checkpoint_fullpath"]}')
        model.on_epoch_end(epoch=epoch)
        logger.info(epoch_progress_meter.get_display_str(epoch+1))
        model.save(epoch=epoch)
        logger.debug(f'model saved in {config["gen_checkpoint_fullpath"]}')
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f'Epoch duration : {epoch_duration:0.2f} seconds '
                    f'or {epoch_duration/60:0.2f} minutes')

    hparams.save(config['params_checkpoint_fullpath'])
    model.on_train_end()

    end_time = time.time()
    total_train_time = end_time - start_time
    logger.info('Training successfully completed')
    logger.info(f'Time taken {total_train_time:0.2f} seconds or '
                f'{total_train_time/60:0.2f} minutes')
    logger.info(str(model.to('cpu')))

    model_path = config['gen_checkpoint_fullpath']
    with open('trained.txt', 'w') as file:
        file.write(model_path)
