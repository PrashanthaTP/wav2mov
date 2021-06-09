"""
Module for creating and saving numoy file from raw dataset
+ mouth landmarks and audio frames are extracted.
"""

import numpy as np
import os


from wav2mov.core.data import RawDataset, GridDataset
from wav2mov.datasets import create_file_list as create_file_list_module

# currently these are not applied anywhere
# audio file in util to see which values are being used instead.
video_frame_rate = 30
audio_sampling_rate = 16_000
win_len = 0.033
win_hop = 0.033



def create(raw_dataset: RawDataset,config,logger) -> str:
    """ Creates numpy file containing video and audio frames from given dataset
           
    Args:
    
    + `raw_dataset` (RawDataset): either GridDataset or Ravdess dataset
    
    + `config` : Config object containing different location information

    Returns:
    
    + `str`: log string containing info about saved file.
    """
   
    samples_count = raw_dataset.samples_count
    dataset_dir = config['train_test_dataset_dir']
    filenames = []
    
    for sample in raw_dataset.generator(get_filepath_only=False,img_size=(256,256),show_progress_bar=True):
        audio_filepath,audio_vals = sample.audio
        _,video_frames = sample.video
        folder_name = os.path.basename(audio_filepath).split('.')[0]
        
    
        if raw_dataset.name in os.path.basename(dataset_dir):
            dest_dir = os.path.join(dataset_dir,folder_name)
        else:
            dest_dir = os.path.join(dataset_dir,f'{raw_dataset.name}',folder_name)
       
        os.makedirs(dest_dir,exist_ok=True)
        np.save(os.path.join(dest_dir,'video_frames.npy'),
                np.array(video_frames))
        np.save(os.path.join(dest_dir,'audio.npy'),
                audio_vals)
        filenames.append(folder_name + '\n')
        
    filenames[-1] = filenames[-1].strip('\n')
    with open(os.path.join(dataset_dir, 'filenames.txt'),'a+') as file:
        file.writelines(filenames)
        
    log = f'Samples generated : {samples_count}\n'
    log += f'Location : { dataset_dir }\n'
    log += f'Filenames are listed in filenames.txt\n'
    logger.info(log)
    create_file_list_module.main()
    logger.debug('train and test filelists created')
    
    return log


def create_from_grid_dataset(config,logger):
    dataset = GridDataset(config['grid_dataset_dir'],
                          audio_sampling_rate=audio_sampling_rate,
                          video_frame_rate=video_frame_rate,
                          samples_count=125)

    create(dataset,config,logger)
    print(f'{dataset.__class__.__name__} successfully processed.')
    # logger.info(log)


if __name__ == '__main__':
    print('No task performed.')
