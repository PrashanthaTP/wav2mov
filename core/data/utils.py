import numpy as np
import torch
from torch.functional import Tensor

import os
import wave
import cv2 
import librosa
import dlib
from collections import namedtuple
from moviepy.editor import AudioFileClip

from wav2mov.core.utils.logger import get_module_level_logger
logger = get_module_level_logger(__name__)

Sample = namedtuple('Sample', ['audio', 'video'])
SampleWithFrames = namedtuple('SampleWithFrames',['audio','audio_frames','video'])

face_detector = dlib.get_frontal_face_detector()

def convert_and_trim_bb(image, rect):
    """ from pyimagesearch
    https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
    """
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    start_x = rect.left()
    start_y = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - start_x
    h = endY - start_y
    # return our bounding box coordinates
    return (start_x, start_y, w, h)

def get_video_frames(video_path,img_size:tuple):
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
            try:
                #image[top_row:bottom_row,left_column:right_column]
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#other libraries including matplotlib,dlib expects image in RGB
                face = face_detector(image)[0]#get first face object
                x,y,w,h = convert_and_trim_bb(image,face)
                image = cv2.resize(image[y:y+h,x:x+w],img_size,interpolation=cv2.INTER_CUBIC)
            except Exception as e:#arises mostly because face_detector could not find the face and resize cannot be done
                h,w,c = image.shape
                if h>0 and w>0:
                  image = cv2.resize(image,img_size,interpolation=cv2.INTER_CUBIC)
                else:
                  raise(e)
            finally:
                frames.append(image)
        return frames 
    except Exception as e:
        logger.error(f'error in getting video frames | filename : {video_path} : {e}')
        
def get_audio(audio_path,sr=None):
    audio,_ = librosa.load(audio_path,sr=sr)#sr=None to get native sampling rate
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
    def __init__(self,audio_sf,coarticulation_factor,stride,device='cpu'):
        self.coarticulation_factor = coarticulation_factor
        self.stride = stride
        self.device = device
        self.audio_sf = audio_sf
        self.n_mfcc = 13
        
    def get_mfccs(self,audio):
        return librosa.feature.mfcc(audio.squeeze().numpy(),sr=self.audio_sf,n_mfcc=self.n_mfcc)[1:].T 
    
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

    def get_audio_frames(self,audio,num_frames=None,get_mfccs=False):
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
            
        possible_num_frames = audio.shape[-1]//self.stride
        num_frames = possible_num_frames if num_frames is None else num_frames
        
        if num_frames > possible_num_frames:
            raise ValueError(f'given audio has {possible_num_frames} frames but {num_frames} frames requested.')
        start_idx = (possible_num_frames-num_frames)//2
        end_idx = (possible_num_frames+num_frames)//2 #start_idx + (num_frames) 
        padding = torch.zeros((1,self.coarticulation_factor*self.stride),device=self.device) 
        audio = torch.cat([padding,audio,padding],dim=1)
        full_mfccs = librosa.feature.mfcc(audio.squeeze().numpy(),sr=self.audio_sf)
        mean,std = np.mean(full_mfccs),np.std(full_mfccs)
        if get_mfccs:
            frames = [self.get_frame_from_idx(audio,idx) for idx in range(start_idx,end_idx)]
            frames = [get_mfccs(frame) for frame in frames]# each of shape [t,13]
            frames = [((frame-mean)/(std+1e-7)) for frame in frames]
            return torch.from_numpy(np.stack(frames,axis=0))# 1,num_frames,(t,13)
        frames = [self.get_frame_from_idx(audio,idx) for idx in range(start_idx,end_idx)]
        #each frame is of shape (1,frame_size) so can be catenated along zeroth dimension .
        return torch.cat(frames,dim=0)
                    
    def get_limited_audio(self,audio,num_frames,start_frame=None,get_mfccs=False) :
        possible_num_frames = audio.shape[-1]//self.stride
        if num_frames>possible_num_frames:
            logger.error(f'Given num_frames {num_frames} is larger the possible_num_frames {possible_num_frames}')

        padding = torch.zeros((audio.shape[0],self.coarticulation_factor*self.stride),device=self.device) 
        audio = torch.cat([padding,audio,padding],dim=1)
            
        # possible_num_frames = audio.shape[-1]//self.stride
        actual_start_frame = (possible_num_frames-num_frames)//2
        # [......................................................]
        #         [................................]
        #         |<-----num_frames---------------->|
        #.........^
        #   actual start frame
        if start_frame is None:
            start_frame = actual_start_frame
            
        if start_frame+num_frames>possible_num_frames:#[why > not >=]think if possible num_frames is 50 and 50 is the required num_frames and start_frame is zero
            logger.warning(f'Given Audio has {possible_num_frames} frames. Given starting frame {start_frame} cannot be consider for getting {num_frames} frames. Changing startframes to {actual_start_frame} frame.')
            start_frame = actual_start_frame
            
        end_frame = start_frame + (num_frames) #exclusive
        
        start_pos = self.__get_center_idx(start_frame)
        end_pos = self.__get_center_idx(end_frame-1)

        full_mfccs = get_mfccs(audio)
        mean,std = np.mean(full_mfccs),np.std(full_mfccs)

        audio = audio[:,self.__get_start_idx(start_pos):self.__get_end_idx(end_pos)]
        if get_mfccs:
            mfccs = list(map(get_mfccs,audio))
            mfccs = [(mfcc-mean/(std+1e-7)) for mfcc in mfccs]
            return torch.from_numpy(np.stack(mfccs,axis=0)) 
        return audio
