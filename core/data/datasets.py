import numpy as np
import math 
import os
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from wav2mov.core.utils.audio import AudioUtil

from wav2mov.core.data.utils import Sample


# TODO : using MFCCS
class AudioVideoDataset(Dataset):
    """Dataset Class for the numpy file containing mouth landamarks and corresponding audio frames.
    """
    def __init__(self,root_dir,
                 filenames_text_filepath,
                 audio_sf,
                 video_fps,
                 num_videos,
                 transform=None):
        self.root_dir = root_dir
        self.audio_fs = audio_sf
        self.video_fps = video_fps
        self.stride = math.floor(self.audio_fs / self.video_fps)
        self.transform = transform
        self.filenames = []
        with open(filenames_text_filepath,'r') as file: 
            self.filenames = file.read().split('\n')
        self.filenames = self.filenames[:num_videos]
        
    def __len__(self):
        return len(self.filenames)
    
    def __load_from_np(self,path):
        return np.load(path)
    
    def __get_folder_name(self,curr_file):
        return os.path.join(self.root_dir,curr_file)
    
    def get_audio(self,idx):
        folder = self.__get_folder_name(self.filenames[idx])
        audio_filepath =  os.path.join(folder,'audio.npy')
        return self.__load_from_np(audio_filepath)
    
    def get_video_frames(self,idx):
        folder = self.__get_folder_name(self.filenames[idx])
        video_filepath =  os.path.join(folder,'video_frames.npy')
        return self.__load_from_np(video_filepath)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        audio = self.get_audio(idx)
        
        # audio = audio.reshape(1,audio.shape[0])
    
        # audio = self.__frame_wise_split(audio)
        video = self.get_video_frames(idx)
        sample = Sample(torch.from_numpy(audio).float(),torch.from_numpy(video).float())
        
        if self.transform:
            sample = self.transform(sample)
       
   
        return sample
    
    def __pad_audio(self,audio):
        padding = torch.tensor([0]*self.stride)
        return torch.cat([padding,audio,padding],dim=0)
        
    def __frame_wise_split(self,audio,frame_num):
        #Zero based indexing for zeroth frame the actual frame is at 0....stride-1[actual_frame]....
        #                                                                   -stride<----------->+2*stride
        # total length of the result is 3*stride
        curr_idx = frame_num*self.stride-1 
        return audio[curr_idx-self.stride:curr_idx+2*self.stride]
        
  
        



    
