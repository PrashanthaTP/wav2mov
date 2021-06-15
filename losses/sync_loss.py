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
        if real_label is None:
            real_label = round(random.uniform(0.8,1),2)
        self.register_buffer('real_label',torch.tensor(real_label))
        self.register_buffer('fake_label',torch.tensor(fake_label))
        self.loss = nn.BCELoss()
        #why not BCELossWithLogits ? (which has sigmoid layer at the entrance)
        # Its because loss is being calculated with respect to cosine distance which is already in the range(0,1)
        # Also using bceloss puts larger penalty while the one having inbuilt sigmoid results in smoother loss
        self.device = device
        
    def get_target_tensor(self,preds,is_real_target):
        target_tensor = self.real_label if is_real_target else self.fake_label
        return target_tensor.expand_as(preds).to(self.device)
    
    def forward(self,preds,is_real_target):
        # preds = nn.functional.cosine_similarity(audio_embedding,video_embedding)
        target_tensor = self.get_target_tensor(preds,is_real_target)
        loss = self.loss(preds,target_tensor)
        logger.debug(f'[sync] loss : {is_real_target} frames  | {loss.item():0.4f}')
        return loss
    