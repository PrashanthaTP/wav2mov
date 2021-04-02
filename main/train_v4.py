""" Training script : can start from previous checkpoint"""

import time


from wav2mov.models.wav2mov_v4 import Wav2MovBW
from wav2mov.utils.audio import StridedAudio

from wav2mov.main.utils import (
                        get_meters,
                        get_device,
                        get_transforms,
                        get_train_dl,
                        add_to_board,
                        get_tensor_logger)

DISPLAY_EVERY = 5
STILL_IMAGE_IDX = 5


def process_sample(sample,hparams):

    img_channels = hparams['img_channels']
    img_size = hparams['img_size']
    transforms = get_transforms((img_size, img_size), img_channels)

    stride = hparams['data']['audio_sf']//hparams['data']['video_fps']
    strided_audio = StridedAudio(stride=stride, coarticulation_factor=0)
    device = get_device(hparams)
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
    bsize,frames,channels,height,width = video.shape
    
    video = video.reshape(bsize*frames,channels,height,width)
    #if not done the `Resize transform` gets angry that it got 3d images 
    # while it was assured previously that it will get 2d images
    video = transforms(video)
    video = video.reshape(bsize,frames,video.shape[-3],video.shape[-2],video.shape[-1])
    
    return get_framewise_audio,video,limit

def train_model(options, hparams, config, logger):
    train_dl, mean, std = get_train_dl(options,config, hparams)

    num_epochs = hparams['num_epochs']
    accumulation_steps = hparams['data']['batch_size']//hparams['data']['mini_batch_size']

    num_batches = len(train_dl)//accumulation_steps
    logger.info(f'option : num_videos : {options.num_videos} | mean : {mean} std: {std} ')
    logger.info(f'len of train dataloader {len(train_dl)}')
    logger.info(f'num_batches : {num_batches}')
    
    start_epoch = 0
    model = Wav2MovBW(config, hparams, logger)
    
    if getattr(options, 'model_path', None) is not None:
        logger.debug(f'Loading pretrained weights : {config.version}')

        prev_epoch = model.load(checkpoint_dir=options.model_path)
        if prev_epoch is not None:
            start_epoch = prev_epoch+1
        logger.debug(f'weights loaded successfully: {config.version}')


    ################################
    # Setup loggers and loss meters
    ################################
    tensor_logger = get_tensor_logger(options, config)
    batch_loss_meters, epoch_loss_meters, epoch_progress_meter,batch_progress_meter= get_meters(hparams,num_batches=num_batches)


    logger.info( f'{STILL_IMAGE_IDX}th frame from last of every video is considered as reference image for the generator')
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
            
            get_framewise_audio,video,limit = process_sample(sample,hparams)
            still_image = video[:, -STILL_IMAGE_IDX, :, :, :]

            model.set_condition(still_image)
            model.on_batch_start()#reset frames history
            for idx in range(limit):
                video_frame = video[:, idx, ...]  # ellipsis
                audio_frame, _ = get_framewise_audio(idx)
               
                model.set_input(audio_frame, video_frame)
                losses = model.optimize_parameters()
                # losses of num_frames * mini_batch
                batch_loss_meters.update(losses, n=video.shape[0]*video.shape[1])
                
            steps += 1
            losses = model.optimize_sequence(real_frames=video[:,:limit,...])
            
            batch_duration += time.time() -batch_start_time

            # losses of mini_batch
            batch_loss_meters.update(losses, n=video.shape[0])
           
            if (batch_idx+1) %accumulation_steps==0:
                model.batch_descent()
                
                # for every video as batch size is 1
                if tensor_logger is not None:
                    add_to_board(tensor_logger, batch_loss_meters.average(),
                                steps, scalar_type='loss')


                epoch_loss_meters.update(batch_loss_meters.average(), n=hparams['data']['batch_size'])
               
                logger.debug(batch_progress_meter.get_display_str((batch_idx+1)//hparams['data']['batch_size']))
                logger.debug(f"Batch duration : {batch_duration:0.4f} seconds or {batch_duration/60:0.4f} minutes")
                
                batch_loss_meters.reset()
                batch_duration = 0.0
      
            if (batch_idx+1) % 5 == 0:
                model.save(epoch=epoch)
                logger.debug('model saved.')

        logger.info(epoch_progress_meter.get_display_str(epoch+1))
        model.save(epoch=epoch)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info( f'Epoch duration : {epoch_duration:0.2f} seconds '
                    f'or {epoch_duration/60:0.2f} minutes')

    hparams.save(config['params_checkpoint_fullpath'])
    model.on_train_end()

    end_time = time.time()
    total_train_time = end_time - start_time
    logger.info('Trainging successfully completed')
    logger.info(
        f'Time taken {total_train_time:0.2f} seconds or {total_train_time/60:0.2f} minutes')
    logger.info(str(model.to('cpu')))

    model_path = config['gen_checkpoint_fullpath']
    with open('trained.txt', 'w') as file:
        file.write(model_path)
