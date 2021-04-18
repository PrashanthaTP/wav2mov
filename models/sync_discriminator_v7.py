from logging import addLevelName
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
    >>>     video_embeddings = self.disc(audio_frame)
    >>>     video_embeddings = torch.cat([video_embeddings.reshape(batch_size,-1),self.desc_v(video_frame).reshape(batch_size,-1)],dim=1)
    >>>     return self.fc(video_embeddings)

    """

    def __init__(self, hparams,use_bias=True):
        super().__init__()
        self.hparams = hparams 
        #5 frames of 666 :3330
        self.disc_a = nn.Sequential(
            nn.Conv1d(1, 64, 494, 50,bias=use_bias),#input 5994 output (5994-494+0)/50 + 1 = 111
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, 1,bias=use_bias),#((111-4+0)/1)+1 =108
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256,4,2,1),#((108-4+2)/2)+1 = 53 +1 = 54
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256,4,2),#((54-4)/2)+1 = 25 + 1 = 26
            nn.ReLU(),
            )
        self.audio_fc = nn.Sequential(nn.Linear(26*256,5*256),
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
        print(f'inside sync forward : audio : {audio_frames.shape} {audio_frames.is_cuda} : video {video_frames.shape} {video_frames.is_cuda}') 
        
        img_height = video_frames.shape[-2]
        video_frames = video_frames[...,img_height//2:,:] #consider only lower half of the image so new height is 256/2 = 128
        
        video_embeddings = self.pre_disc_v(video_frames)#out has no Temporal(i,e where frames are stacked) dimension | 3d (F,H,W) to 2d(H,W)
        video_embeddings = video_embeddings.reshape(batch_size,
                                                    video_embeddings.shape[1],
                                                    video_embeddings.shape[-2],
                                                    video_embeddings.shape[-1])
        video_embeddings = self.disc_v(video_embeddings).reshape(batch_size, -1)
        video_embeddings = self.video_fc(video_embeddings)
        
        audio_frames = audio_frames.reshape(batch_size, 1, -1)
    
        # print(f'audio frames {audio_frames.shape} self.disc_a(audio_frames) {self.disc_a(audio_frames).shape}')

        audio_embeddings = self.disc_a(audio_frames).reshape(batch_size,-1)
        audio_embeddings = self.audio_fc(audio_embeddings)
        # print(f'audio frames {audio_frames.shape}') 
        #see syncnet forward pass https://github.com/Rudrabha/Wav2Lip/blob/master/models/syncnet.py#L62-L63
        audio_embeddings = nn.functional.normalize(audio_embeddings,p=2,dim=1) 
        video_embeddings = nn.functional.normalize(video_embeddings,p=2,dim=1) 
        # print(self.disc_a(audio_frames).shape,audio_frames.shape)
        return audio_embeddings,video_embeddings 

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))
