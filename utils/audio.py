import math
import torch 

class StridedAudio:
    def __init__(self,stride,coarticulation_factor=0):
        self.coarticulation_factor = coarticulation_factor 
        self.stride = stride 
        if isinstance(self.stride,float):
            self.stride = math.floor(self.stride)
        
    
    def get_frame_wrapper(self,audio):
        # padding = torch.tensor([])
        # audio = torch.from_numpy(audio)
        
        if self.coarticulation_factor!=0:
            padding = torch.cat([torch.tensor([0]*self.stride) for _ in range(self.coarticulation_factor)],dim=0)
            audio = torch.cat([padding,audio,padding])
            
            audio = audio.unsqueeze(0)
            
        def get_frame(idx):
            center_frame_pos = idx*self.stride
            # print(center_frame_pos)
            center_frame = audio[:,center_frame_pos:center_frame_pos+self.stride]
            if self.coarticulation_factor == 0:
                return center_frame, center_frame_pos+self.stride
            # curr_idx = center_frame_pos
            # print(f'audio shape {audio.shape} centerframe {center_frame.shape}')
            for i in range(self.coarticulation_factor):
                center_frame = torch.cat([audio[:,center_frame_pos-(i-1)*self.stride:center_frame_pos-i*self.stride],center_frame],dim=1)
            last_pos = 0
            for i in range(self.coarticulation_factor):
                center_frame = torch.cat([audio[:,center_frame_pos+(i+1)*self.stride:center_frame_pos+(i+2)*self.stride]])
                last_pos = center_frame_pos+(i+2)*self.stride
            return center_frame,last_pos
        return get_frame
    
    
class StridedAudioV2:
    def __init__(self, stride, coarticulation_factor=0,device='cpu'):
        """ 
        0.15s window ==>16k*0.15 points = 2400
        so padding on either side will be 2400//2 = 1200
        """
        self.coarticulation_factor = coarticulation_factor
        self.stride = stride
        if isinstance(self.stride, float):
            self.stride = math.floor(self.stride)
        self.pad_len = self.coarticulation_factor*self.stride
        self.device = device
        
    def get_frame_wrapper(self, audio):
        # padding = torch.tensor([])
        # audio = torch.from_numpy(audio)

        if self.coarticulation_factor != 0:
         
            padding = torch.zeros((audio.shape[0],self.pad_len),device=self.device)
            
            # padding = torch.cat([torch.tensor([0]*self.stride)
            #                      for _ in range(self.coarticulation_factor)], dim=0)
          
            audio = torch.cat([padding, audio,padding],dim=1)
            # print(audio.shape,self.padding.shape)
    
        
        def get_frame_from_idx(idx):
            center_idx= (idx) + (self.coarticulation_factor)
            start_pos = (center_idx-self.coarticulation_factor)*self.stride
            end_pos = (center_idx+self.coarticulation_factor+1)*self.stride
            return audio[:,start_pos:end_pos]
    
        def get_frames_from_range(start_idx,end_idx):
            start_pos = start_idx + self.coarticulation_factor
            start_idx  = (start_pos-self.coarticulation_factor)*self.stride
            end_pos = end_idx+self.coarticulation_factor
            end_idx = (end_pos+self.coarticulation_factor+1)*self.stride
            return audio[:,start_idx:end_idx]
        
        return get_frame_from_idx,get_frames_from_range
    
            
            
