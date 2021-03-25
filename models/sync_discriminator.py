import torch
from torch import nn, optim

from wav2mov.core.models.base_model import BaseModel
# Audio and current video_frame image


class SyncDiscriminator(BaseModel):
    """
    >>> self.desc_a = nn.Sequential(
    >>>         nn.Conv1d(1, 4, 4, 2, 1),
    >>>         nn.ReLU(),
    >>>         nn.Conv1d(4, 1, 4, 2, 1),
    >>>         nn.ReLU()
    >>>         )
    >>>
    >>> self.desc_v = nn.Sequential(
    >>>         nn.Conv2d(3,6,4,2,1),
    >>>         nn.BatchNorm2d(6),
    >>>         nn.ReLU(),
    >>>         nn.Conv2d(6,32,4,2,1),
    >>>         nn.BatchNorm2d(32),
    >>>         nn.ReLU(),
    >>>         nn.Conv2d(32,64,4,2,1),
    >>>         nn.BatchNorm2d(64),
    >>>         nn.ReLU(),
    >>>         # nn.Conv2d(64,128,4,2,1),
    >>>         # nn.BatchNorm2d(128),
    >>>         # nn.ReLU(),
    >>>         nn.Conv2d(64,1,4,2,1),
    >>>         nn.ReLU()
    >>>
    >>>         )
    >>>
    >>> self.fc = nn.Sequential(
    >>>         nn.Linear(166+16*16,256),
    >>>         nn.ReLU(),
    >>>         nn.Linear(256,128)
    >>>         )

    >>> def forward(self, audio_frame,video_frame):
    >>>
    >>>     batch_size = audio_frame.shape[0]
    >>>     audio_frame = audio_frame.reshape(batch_size,1,-1)
    >>>     x = self.desc_a(audio_frame)
    >>>     x = torch.cat([x.reshape(batch_size,-1),self.desc_v(video_frame).reshape(batch_size,-1)],dim=1)
    >>>     return self.fc(x)

    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams 
        self.desc_a = nn.Sequential(
            nn.Conv1d(1, 64, 3, 3,bias=False),#input 666 output (666-3+0)/3 + 1 = 663/3 + 1 = 222
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, 3,bias=False),#((222-3+0)/3)+1 = 219/3 +1 =74
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 4,4, 2, 1),#((74-4+2)/2)+1 = 36 +1 = 37
        
            nn.ReLU()
            )

        # height and width are halved in each step
        self.desc_v = nn.Sequential(
            nn.Conv2d(self.hparams['in_channels'], 6, 4, 2, 1,bias=False), # height_new = ((128-4+2)/2)+1 = 126/2 + 1 = 64
            nn.BatchNorm2d(6),                                # width_new = 127+1 = 128
            nn.LeakyReLU(0.2),  
            nn.Conv2d(6, 32, 4, 2, 1,bias=False), # #H_new = 32, W_new = 64
            nn.BatchNorm2d(32),       
            nn.LeakyReLU(0.2),  
            nn.Conv2d(32, 64, 4, 2, 1,bias=False), #H_new,W_new = 16,32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 4, 2, 1),  #H_new,W_new = 8,16
            nn.LeakyReLU(0.2)

        )

        self.fc = nn.Sequential(
            nn.Linear(4*37+8*16, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
            )

    def forward(self, audio_frame, video_frame):
        """audio_frame is of shape : [batch_size,1,N] and video_frame is of shape : [batch_size,channels,height,width]

        """
        batch_size = audio_frame.shape[0]
        audio_frame = audio_frame.reshape(batch_size, 1, -1)
        img_height = video_frame.shape[-2]
        video_frame = video_frame[...,img_height//2:,:] #consider only lower half of the image so new height is 256/2 = 128
        # print(audio_frame.shape,video_frame.shape)
        x = self.desc_a(audio_frame)
        # print(x.shape,(video_frame).shape)
        
        x = torch.cat([x.reshape(batch_size, -1),
                      self.desc_v(video_frame).reshape(batch_size, -1)], dim=1)
        return self.fc(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))
