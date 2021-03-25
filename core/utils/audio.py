import numpy as np
import cupy as cp
import librosa
from moviepy.editor import AudioFileClip



class AudioUtil:
    '''provides methods for loading audio file and extracting frames/mfccs'''
    def __init__(self,use_cupy:bool,logger=None):
        """
        Arguments:
            use_cupy {bool} -- if True uses `cupy` instead of `numpy`. Requires cuda to be installed on the machine.
        """
        if use_cupy:
            self.cp = cp 
        else:
            self.cp = np 
        log_str = f'Using {self.cp.__name__} for {self.__class__.__name__} ops.'
        if logger is None:
            print(log_str)
        else:
            logger.info(log_str)
        
    @staticmethod
    def get_audio_from_video(video_filepath):
        temp_path = 'temp_audio.wav'
        AudioFileClip(video_filepath).write_audiofile(temp_path,verbose=False,logger=None)
        sig,fs = AudioUtil.load_audio(temp_path)
        return sig,fs

    @staticmethod
    def load_audio(audio_filepath):
        sig,fs =  librosa.load(audio_filepath,sr=None)
        return sig,fs
    
    def get_audio_frames(self,
                        file,
                        frame_len_sec = 0.033,
                        is_video=False):
        '''
        -----------------------------
        Splits the given video into frames of given length
        ---------------
        
        Parameters
        ---------------
        + file : 
            Either audio or video(is_video should be given as True)
        + frame_len_sec
            Length of each audio frame in seconds ,Default 0.033s
        + is_video
            If file is video,this parameter should be passed as True,Default False
            
        Returns (tuple)
        ---------------
        + Frames [numpy array]
        + Samples in each frames [int]
        + Count of frames [int]
        
        '''
        load_signal = AudioUtil.load_audio
        if is_video:
            load_signal = AudioUtil.get_audio_from_video
   
        sig,fs = load_signal(file)
        
        frames_arr,each_frame_len = self.get_frames(sig,fs,frame_len_sec)
        padded_frames_arr = frames_arr
        if len(frames_arr[-1])<each_frame_len:
            self.pad_signal(frames_arr,each_frame_len)#inplace operation 
        return padded_frames_arr,each_frame_len,len(frames_arr)
    
    def get_frames(self,sig,fs,frame_len_sec):
        frame_len = int(frame_len_sec * fs)
        frames_arr = [sig[i:i+frame_len] for i in range(0,len(sig),frame_len)]
        return frames_arr,frame_len
    
    def pad_signal(self,frames,frame_len):
        mean_val = self.cp.mean(frames[-1])
        pad = self.cp.tile(mean_val,frame_len-len(frames[-1]))
        frames[-1] = self.cp.hstack([frames[-1],pad])
     
    @staticmethod    
    def extract_mfccs(X,sample_rate=44100):
        """Returns mffcs coefficients shape :(40,)"""
        # stft = np.abs(librosa.stft(X))
        mfccs = cp.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    
        return mfccs
    