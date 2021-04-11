import torch
from torch import nn, optim

from wav2mov.core.models.base_model import BaseModel
# Audio and current video_frame image


class SyncDiscriminator(BaseModel):
    """
    >>> self.disc = nn.Sequential(
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
    >>>     x = self.disc(audio_frame)
    >>>     x = torch.cat([x.reshape(batch_size,-1),self.desc_v(video_frame).reshape(batch_size,-1)],dim=1)
    >>>     return self.fc(x)

    """

    def __init__(self, hparams,use_bias=True):
        super().__init__()
        self.hparams = hparams 
        #5 frames of 666 :3330
        self.disc_a = nn.Sequential(
            # nn.Conv1d(1, 64, 5,1,1,bias=use_bias),#input 666 output (666-5+2)/1 = 663
            # nn.ReLU(),
            # nn.Conv1d(64, 128, 3, 3,bias=use_bias),#((663-3+0)/3)+1 = 660/3 +1 =221
            # nn.ReLU(),
            # nn.Conv1d(128, 512,3,1,1),#((221-3+2)/2)+1 = 220 +1 = 221
            # nn.ReLU(),
            # nn.Conv1d(512,1,3, 1),#((221-3)/1)+1 = 219
            # nn.ReLU(),
            nn.Conv1d(1, 64, 330, 30,bias=use_bias),#input 666 output (3330-330+0)/30 + 1 = 101
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, 1,bias=use_bias),#((101-3+0)/1)+1 =98
            nn.ReLU(),
            nn.Conv1d(128, 256,4,2,1),#((98-4+2)/2)+1 = 48 +1 = 49
            nn.ReLU(),
        
        
            )
        self.audio_fc = nn.Sequential(nn.Linear(49*256,5*256),
                                      nn.ReLU(),
                                      nn.Linear(5*256,256),
                                      nn.ReLU())
        # height and width are halved in each step
        self.pre_disc_v = nn.Sequential(
            nn.Conv3d(self.hparams['in_channels'], 32, (5,4,4), (1,2,2),(0,1,1), bias=use_bias),
            nn.LeakyReLU(0.2))

        ##############################################
        #pre_disc_V : F :5 ==> ((5-5)+0)/1) + 1 =>1 
        #            H : 128 ==>((128-4+2)/2) + 1 =>64
        #           W : 256 ==>((256-4+2)/2) + 1 =>128
        # out is of shape : 32x64x128
        #############################################
        self.disc_v = nn.Sequential( 
            nn.Conv2d(32, 64, 4, 2, 1,bias=use_bias), # #H_new = 32, W_new = 64     
            nn.LeakyReLU(0.2),  
            nn.Conv2d(64, 128, 4, 2, 1,bias=use_bias), #H_new,W_new = 16,32
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  #H_new,W_new = 8,16
            nn.LeakyReLU(0.2)

        )
        self.video_fc = nn.Sequential(nn.Linear(256*8*16,256*8),
                                      nn.ReLU(),
                                      nn.Linear(256*8,256),
                                      nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(256+256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
            )

    def forward(self, audio_frames, video_frames):
        """audio_frames is of shape : [batch_size,1,N] and video_frames is of shape : [batch_size,channels,height,width]

        """
        batch_size = audio_frames.shape[0]
        
        
        img_height = video_frames.shape[-2]
        video_frames = video_frames[...,img_height//2:,:] #consider only lower half of the image so new height is 256/2 = 128
        
        x = self.pre_disc_v(video_frames)
        x = x.reshape(batch_size,x.shape[1],x.shape[-2],x.shape[-1])
        x = self.disc_v(x).reshape(batch_size, -1)
        x = self.video_fc(x)
        
        audio_frames = audio_frames.reshape(batch_size, 1, -1)
        # print(self.disc_a(audio_frames).shape,audio_frames.shape)
        x = torch.cat([self.audio_fc(self.disc_a(audio_frames).reshape(batch_size, -1)),
                      x], dim=1)
        return self.fc(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))
