import random
import torch
from torch import nn

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)

class SyncLoss(nn.Module):
    """Abstracts away the funciton of applying loss of synchronity between audio and video frames.
    [Reference]:
    https://github.com/Rudrabha/Wav2Lip/blob/master/hq_wav2lip_train.py#L181-L186 
    """
    def __init__(self,device,real_label=None,fake_label=0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        # self.register_buffer('real_label',torch.tensor(real_label))
        # self.register_buffer('fake_label',torch.tensor(fake_label))
        self.loss = nn.BCELoss()
        self.device = device
        
    def get_target_tensor(self,preds,is_real_target):
        real_label = torch.tensor( round(random.uniform(0.8,1),2) if self.real_label is None else self.real_label )
        target_tensor =  if is_real_target else torch.tensor(self.fake_label)
        return target_tensor.expand_as(preds).to(self.device)
    
    def forward(self,preds,is_real_target):
        # preds = nn.functional.cosine_similarity(audio_embedding,video_embedding)
        target_tensor = self.get_target_tensor(preds,is_real_target)
        loss = self.loss(preds,target_tensor)
        logger.debug(f'[sync] loss {is_real_target} frames :{loss.item():0.4f}')
        return loss
