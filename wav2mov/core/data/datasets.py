import numpy as np
import math 
import os
import torch
from torch.utils.data import Dataset
from collections import namedtuple

from wav2mov.core.data.utils import Sample


class AudioVideoDataset(Dataset):
    """
    Dataset for audio and video numpy files 
    """
    def __init__(self,
                 root_dir,
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
        video = self.get_video_frames(idx)
        audio = torch.from_numpy(audio)
        video = torch.from_numpy(video).permute(0,3,1,2)#F,H,W,C ==> F,C,H,W
        video = video/255
        sample = Sample(audio,video)
        if self.transform:
            sample = self.transform(sample)
        return sample
        
  
        
    
