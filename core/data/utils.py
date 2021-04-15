import logging 
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


def get_video_frames(video_path,img_size):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if(not cap.isOpened()):
            logging.error("Cannot open video stream or file!")
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
        logging.error(f'error in getting video frames | filename : {video_path}')
        
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
    def __init__(self,coarticulation_factor,stride):
        self.coarticulation_factor = coarticulation_factor
        self.stride = stride

    def get_frame_from_idx(self,audio,idx):
        if not isinstance(audio,Tensor):
            audio = torch.tensor(audio)
            
        if len(audio.shape)<2:
            audio = audio.unsqueeze(0)
            
        center_idx = (idx) + (self.coarticulation_factor)
        start_pos = (center_idx-self.coarticulation_factor)*self.stride
        end_pos = (center_idx+self.coarticulation_factor+1)*self.stride
        return audio[:, start_pos:end_pos]

    def get_audio_frames(audio):
        
        
