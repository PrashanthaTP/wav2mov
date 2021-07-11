import librosa
import torch
from torch import nn
from torch.functional import Tensor
from torchaudio.transforms import MFCC
from inference import params
from inference.utils import get_module_level_logger
logger = get_module_level_logger(__file__)

class AudioUtil:
    def __init__(self,audio_sf,coarticulation_factor,stride,device='cpu'):
        self.coarticulation_factor = coarticulation_factor
        self.stride = stride
        self.device = device
        self.audio_sf = audio_sf
        self.n_mfcc = 14
        self.n_fft = 2048
        self.win_length = self.n_fft
        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 128
        n_mfcc = 14
        sample_rate=16000

        self.mfcc_transform =nn.Sequential(
                                      MFCC(sample_rate=sample_rate,
                                      n_mfcc=n_mfcc,
                                      melkwargs={
                                        'n_fft': n_fft,
                                        'n_mels': n_mels,
                                        'hop_length': hop_length,
                                        'mel_scale': 'htk',
                                      }))

        self.mfcc_transform.to(self.device)
        
    def extract_mfccs(self,audio):
        with torch.no_grad():
            mfccs = self.mfcc_transform(audio.squeeze(0))[1:].T 
            # logger.error(f'mfccs shape : {mfccs.shape}')
        if isinstance(mfccs,torch.functional.Tensor):
          return mfccs
        return torch.from_numpy(mfccs).to(audio.device)
 
    def get_mfccs_mean_std(self,audio):
        full_mfccs = list(map(self.extract_mfccs,audio))
        full_mfccs = torch.stack(full_mfccs,axis=0)#of shape N,T,n_mfcc
        mean,std = torch.mean(full_mfccs,axis=(1),keepdims=True),torch.std(full_mfccs,axis=(1),keepdims=True)
        return mean,std
 
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
        if len(audio.shape)<2:
            audio = audio.unsqueeze(0)
        possible_num_frames = audio.shape[-1]//self.stride
        # logger.debug(possible_num_frames)
        num_frames = possible_num_frames if num_frames is None else num_frames
        
        mean,std = self.get_mfccs_mean_std(audio)
      
        if num_frames > possible_num_frames:
            raise ValueError(f'given audio has {possible_num_frames} frames but {num_frames} frames requested.')
        start_idx = (possible_num_frames-num_frames)//2
        end_idx = (possible_num_frames+num_frames)//2 #start_idx + (num_frames) 
        padding = torch.zeros((1,self.coarticulation_factor*self.stride),device=self.device) 
        audio = torch.cat([padding,audio,padding],dim=1)
        if get_mfccs:
            frames = [self.get_frame_from_idx(audio,idx) for idx in range(start_idx,end_idx)]
            frames = [self.extract_mfccs(frame) for frame in frames]# each of shape [t,13]
            # frames = [((frame-mean[i])/(std[i]+1e-7)) for i,frame in enumerate(frames)]
            frames = torch.stack(frames,axis=0)# 1,num_frames,(t,13)
            # logger.warning(f'frames {frames.shape} mean : {mean.shape}')
            return (frames-mean)/(std+1e-7)
        frames = [self.get_frame_from_idx(audio,idx) for idx in range(start_idx,end_idx)]
        #each frame is of shape (1,frame_size) so can be catenated along zeroth dimension .
        return torch.cat(frames,dim=0)
                    
    def get_limited_audio(self,audio,num_frames,start_frame=None,get_mfccs=False) :
        possible_num_frames = audio.shape[-1]//self.stride
        if num_frames>possible_num_frames:
            logger.error(f'Given num_frames {num_frames} is larger the possible_num_frames {possible_num_frames}')
 
        mean,std = self.get_mfccs_mean_std(audio)
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
 
        audio = audio[:,self.__get_start_idx(start_pos):self.__get_end_idx(end_pos)]
        if get_mfccs:
            mfccs = list(map(self.extract_mfccs,audio))
            # mfccs = [(mfcc-mean[i]/(std[i]+1e-7)) for i,mfcc in enumerate(mfccs)]
            mfccs = torch.stack(mfccs,axis=0)
            return (mfccs-mean)/(std+1e-7)
        return audio

def load_audio(audio_path):
    audio,sr = librosa.load(audio_path,sr=params.AUDIO_SF)
    return torch.from_numpy(audio)

def get_num_frames(audio):
    audio_len = audio.squeeze().shape[0]
    return audio_len//params.STRIDE
    
def preprocess_audio(audio):
    if not isinstance(audio,Tensor):
        audio = torch.from_numpy(audio)
    audio = (audio-params.AUDIO_MEAN)/(params.AUDIO_STD + params.EPSILON)
    framewise_mfccs = AudioUtil(params.AUDIO_SF,params.COARTICULATION_FACTOR,
                                params.STRIDE,params.DEVICE).get_audio_frames(audio,get_mfccs=True)
    return framewise_mfccs
