"""Raw Datasets Classes : Grid and Ravdess"""
import os
from abc import abstractmethod
from collections import namedtuple
from os import path
from tqdm import tqdm

from wav2mov.core.data.utils import get_video_frames,get_audio,get_audio_from_video


SampleContainer = namedtuple('SampleContainer',['audio','video'])
Sample = namedtuple('Sample',['path','val'])



class RawDataset:
    """
    Description of RawDataset

    Args:
        location (undefined):
        audio_sampling_rate=None (undefined):
        video_frame_rate=None (undefined):
        website=None (undefined):

    """
    
    
    def __init__(self,
                 location,
                 audio_sampling_rate=None,
                 video_frame_rate=None,
                 samples_count = None,
                 website=None):
        """
        Description of __init__

        Args:
            self (undefined):
            location (undefined):
            audio_sampling_rate=None (undefined):
            video_frame_rate=None (undefined):
            samples_count=None
            website=None (undefined):

        """
        self.root_location = location
        self.audio_sampling_rate = audio_sampling_rate
        self.video_frame_rate = video_frame_rate
        self.samples_count = samples_count 
        self.website = website
        
    def info(self):
        return print(f'self.__class__.__name__ : {self.website}')
    
    # def create_numpy_files(self):
    #     raise NotImplementedError("function create_numpy_file is not implemented.")
    @abstractmethod
    def generator(self):
        raise NotImplementedError(f'{self.__class__.__name__} should implement generator method')
    
class RavdessDataset(RawDataset):
   
    """Dataset class for RAVDESS dataset.
    
    [LINK] https://zenodo.org/record/1188976#.YBlpZOgzZPY

    Folder structure
    ------------------
    
    `
    root_location:folder 
    |
    |___actor1:folder
    |      |
    |      |_ video 1:file
    |      |_ video 2:file
    |      |_ ...
    |
    |___actor2:folder
           |
           |_ video 1:file
           |...
   `
   
    """
    link = r"https://zenodo.org/record/1188976#.YBlpZOgzZPY"
    name = 'ravdess_dataset'
    def __init__(self,location,audio_sampling_rate,video_frame_rate,samples_count=None,link=None):
        super().__init__(location,audio_sampling_rate,video_frame_rate,samples_count,link)
        self.sub_folders = (folder for folder in  os.listdir(location) 
                            if os.path.isdir(os.path.join(location,folder)))
        
       
        
    def generator(self)->Sample:
        """yields audio and video filenames one by one

        Returns:
            Sample: containes audio and video filpaths

        Yields:
            Iterator[Sample]: 
            
        Examples:
        
            >>>dataset = RavdessDataset(...)
            >>>for sample in dataset.generator():
            >>>    audio_filepath = sample.audio
        
        """
        for actor in self.sub_folders:
            actor_path = os.path.join(self.root_location,actor)
            videos = (video for _,_,video in os.walk(actor_path))
            limit = self.samples_count if self.samples_count!=None else len(videos[0])
            for video in tqdm(videos[:limit],
                                  desc="Ravdess Dataset",
                                  total=len(videos),ascii=True,colour="green"):
                audio_path = get_audio_from_video(os.path.join(actor_path,video))
                video_path = os.path.join(actor_path,video)
                yield SampleContainer(video=Sample(video_path,val=None),
                                      audio=Sample(audio_path,val=None))


class GridDataset(RawDataset):
    """Dataset class for Grid dataset.
    
    [LINK] http://spandh.dcs.shef.ac.uk/avlombard/
    
    [LINK] [PAPER]  https://asa.scitation.org/doi/10.1121/1.5042758

    `Folder structure`::
    
    root_location:folder 
    |
    |___audio:folder
    |      |
    |      |_ .wav:file
    |      |_ .wav:file
    |      |_ ...
    |
    |___video:folder
           |
           |_ .mov:file
           |_ .mov:file
           |_ ...
  
    """
    link = "http://spandh.dcs.shef.ac.uk/avlombard/"
    name = 'grid_dataset'
    def __init__(self,location,audio_sampling_rate,video_frame_rate,samples_count=None,link=None):
        super().__init__(location,
                         audio_sampling_rate,
                         video_frame_rate,
                         samples_count,
                         link)

        
    def generator(self,get_filepath_only=True,img_size=128,show_progress_bar=True)->Sample:
        video_folder = os.path.join(self.root_location,'video/')
        audio_folder = os.path.join(self.root_location,'audio/')
        videos = [video for _,_,video in os.walk(video_folder)]
        if self.samples_count is None : self.samples_count = len(videos[0])
        self.samples_count = min(len(videos[0]),self.samples_count)
        limit = self.samples_count if self.samples_count!=None else len(videos[0])
        progress_bar = tqdm(enumerate(videos[0][:limit]),
                            desc="Grid Dataset",
                            total=len(videos[0][:limit]), ascii=True, colour="green") if show_progress_bar else enumerate(videos[0][:limit])
        
        for idx,video_filename in progress_bar:
            audio_filename = video_filename.split('.')[0] + '.wav'
            video_path = os.path.join(video_folder, video_filename)
            audio_path = os.path.join(audio_folder, audio_filename)
    
            video_val = None if get_filepath_only else get_video_frames(video_path, img_size=img_size)
            audio_val = None if get_filepath_only else get_audio(audio_path)
            
            res = SampleContainer(video=Sample(video_path, val=video_val),
                                 audio=Sample(audio_path, val=audio_val))
            # if idx+1==limit:#what if the actual length of videos is less than the limit passed by talaharte hudga
            #     return res 
            # else:
            yield res
   
   
