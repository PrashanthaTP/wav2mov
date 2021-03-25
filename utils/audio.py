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