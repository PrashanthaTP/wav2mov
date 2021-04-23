import torch
import logging 
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
import os
# import dlib
import wave
from moviepy.editor import AudioFileClip
import numpy as np


import cv2 
import librosa 
import imutils  # for image resizing
from imutils import face_utils

from collections import namedtuple

from torch.functional import Tensor


Sample = namedtuple('Sample', ['audio', 'video'])
SampleWithFrames = namedtuple('SampleWithFrames',['audio','audio_frames','video'])

def get_video_frames(video_path,img_size):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if(not cap.isOpened()):
            logger.error("Cannot open video stream or file!")
        frames = []
        while cap.isOpened():
            frameId = cap.get(1)
            ret, image = cap.read()
            if not ret:
                break
            image = imutils.resize(image, width=img_size)
            frames.append(image)
        return frames 
    except Exception as e:
        logger.error(f'error in getting video frames | filename : {video_path}')
        
def get_audio(audio_path):
    audio,_ = librosa.load(audio_path,sr=None)#sr=None to get native sampling rate
    return audio


def get_audio_sampling_rate(wav_file_fullpath):
    with wave.open(wav_file_fullpath, 'rb') as wave_file:
        return wave_file.getFrameRate()


def get_audio_from_video(video_file):
    AudioFileClip(video_file).write_audiofile('temp.wav', verbose=False, logger=None)
    return 'temp.wav'

"""
shape_predictor_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

mouth_pos = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
mouth_start_pos, mouth_end_pos = mouth_pos
"""


class AudioUtil:
    def __init__(self,coarticulation_factor,stride,device='cpu'):
        self.coarticulation_factor = coarticulation_factor
        self.stride = stride
        self.device = device
    def __get_center_idx(self,idx):
        return idx+self.coarticulation_factor
    def __get_start_idx(self,idx):
        return (idx-self.coarticulation_factor)*self.stride
    def __get_end_idx(self,idx):
        return (idx+self.coarticulation_factor+1)*self.stride
    
    def get_frame_from_idx(self,audio,idx):
        if not isinstance(audio,Tensor):
            audio = torch.tensor(audio)
            
        if len(audio.shape)<2:
            audio = audio.unsqueeze(0)
            
        center_idx = self.__get_center_idx(idx)
        start_pos = self.__get_start_idx(center_idx)
        end_pos = self.__get_end_idx(center_idx)
        return audio[:, start_pos:end_pos]

    def get_audio_frames(self,audio,num_frames=None):
        """extracts from the audio      

        Args:
            audio ([numpy array or Tensor]): audio to be seperated into frames (1,audio_points)
            num_frames (int) : required number of frames.If None all possible frames are returned.Defaults to None.

        Returns:
            [Tensor]:stacked  audio frames of shape (1,num_frames)
        
        Raises:
            ValueError : if frange is not valid.
        """
        if not isinstance(audio,Tensor):
            audio = torch.tensor(audio,device=self.device)
        # if len(audio.shape)<2:
        #     audio = audio.unsqueeze(0)
            
        actual_frames = audio.shape[-1]//self.stride
        num_frames = actual_frames if num_frames is None else num_frames
        
        if num_frames > actual_frames:
            raise ValueError(f'given audio has {actual_frames} frames but {num_frames} frames requested.')
        start_idx = (actual_frames-num_frames)//2
        end_idx = (actual_frames+num_frames)//2 #start_idx + (num_frames) 
        padding = torch.zeros((1,self.coarticulation_factor*self.stride),device=self.device) 
        audio = torch.cat([padding,audio,padding],dim=1)

        frames = [self.get_frame_from_idx(audio,idx) for idx in range(start_idx,end_idx)]
        #each frame is of shape (1,frame_size) so can be catenated along zeroth dimension .
        return torch.cat(frames,dim=0)
                    
    def get_limited_audio(self,audio,num_frames,start_frame=None) :
            actual_frames = audio.shape[-1]//self.stride
            if num_frames>actual_frames:
                logger.error(f'Given num_frames {num_frames} is larger the possible_num_frames {actual_frames}')

            padding = torch.zeros((audio.shape[0],self.coarticulation_factor*self.stride),device=self.device) 
            audio = torch.cat([padding,audio,padding],dim=1)
 
            actual_start_frame = (actual_frames-num_frames)//2
            if start_frame is None:
                start_frame = actual_start_frame
                
            if start_frame+num_frames>actual_frames:
                logger.warning(f'Given Audio has {actual_frames} frames. Given starting frame {start_frame} cannot be consider for getting {num_frames}frames. Changing startframes to {actual_start_frame} frame.')
                start_frame = actual_start_frame
                
            end_frame = start_frame + (num_frames) #exclusive
            
            start_pos = self.__get_center_idx(start_frame)
            end_pos = self.__get_center_idx(end_frame-1)
            audio = audio[:,self.__get_start_idx(start_pos):self.__get_end_idx(end_pos)]
            return audio