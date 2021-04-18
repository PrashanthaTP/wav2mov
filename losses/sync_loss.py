import torch
from torch import nn


class SyncLoss(nn.Module):
    """Abstracts away the funciton of applying loss of synchronity between audio and video frames.
    [Reference]:
    https://github.com/Rudrabha/Wav2Lip/blob/master/hq_wav2lip_train.py#L181-L186 
    """
    def __init__(self,device,real_label=1.0,fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label',torch.tensor(real_label))
        self.register_buffer('fake_label',torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()
        self.device = device
        
    def get_target_tensor(self,preds,is_real_target):
        target_tensor = self.real_label if is_real_target else self.fake_label
        return target_tensor.expand_as(preds).to(self.device)
    
    def forward(self,audio_embedding,video_embedding,is_real_target):
        preds = nn.functional.cosine_similarity(audio_embedding,video_embedding)
        target_tensor = self.get_target_tensor(preds,is_real_target)
        return self.loss(preds,target_tensor)