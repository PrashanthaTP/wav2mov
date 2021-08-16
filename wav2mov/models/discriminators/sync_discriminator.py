import torch
from torch import nn, optim

from wav2mov.models.layers.conv_layers import Conv1dBlock,Conv2dBlock
from wav2mov.models.layers.debug_layers import IdentityDebugLayer
from wav2mov.models.utils import get_same_padding

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)

from wav2mov.core.models.base_model import BaseModel
from wav2mov.models.discriminators.utils import save_series

class SyncDiscriminator(BaseModel):
    def __init__(self, hparams,config):
        super().__init__()
        self.hparams = hparams 
        self.config = config
        self.relu_neg_slope = self.hparams['relu_neg_slope'] 
        ##############################################################
        # Audio encoding
        ################################################################
        # 1->32->64->128->256->512
        padding_42 = get_same_padding(kernel_size=4,stride=2)#(4-2)/2 = 1
        padding_31 = get_same_padding(kernel_size=3,stride=1)#(3-1)/2 = 1
        padding_32 = get_same_padding(kernel_size=3,stride=2)#(3-2)/2 = 1
        self.audio_encoder = nn.Sequential(
            Conv2dBlock(1,32,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU()),#12,13
            Conv2dBlock(32,32,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(32,32,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),  
              
            Conv2dBlock(32,64,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU()),    #12,13
            Conv2dBlock(64,64,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(64,64,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),  
              
            Conv2dBlock(64,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU()),    #12,13
            Conv2dBlock(128,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(128,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
          
            Conv2dBlock(128,256,(4,3),(2,1),padding=1,use_norm=True,use_act=True,act=nn.ReLU()),  #6,13
            Conv2dBlock(256,256,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(256,256,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
         
            Conv2dBlock(256,512,(4,3),(2,1),padding=1,use_norm=True,use_act=True,act=nn.ReLU()),  #3,13
            Conv2dBlock(512,512,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(512,512,(3,5),(1,2),padding=(0,1),use_norm=True,use_act=True,act=nn.ReLU()),#1,6,
            )
        
        self.audio_fc = nn.Sequential(
            nn.Linear(6*512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU()
            ) 
        # 3=>3d cnn => 32 ->64->128->256->512
        self.frames_3d_to_2d = nn.Sequential(
            nn.Conv3d(self.hparams['in_channels'], 32, (5,4,4), (1,2,2),(0,1,1) ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # height and width are halved in each step
            ##############################################
            #pre_disc_V : F :5 ==> ((5-5)+0)/1) + 1 =>1 
            #            H : 128 ==>((128-4+2)/2) + 1 =>64
            #           W : 256 ==>((256-4+2)/2) + 1 =>128
            # out is of shape : 32x64x128
            #############################################
            ) 
        self.frames_encoder = nn.Sequential(
            Conv2dBlock(32,64,4,2,padding_42,use_norm=True,use_act=True,act=nn.ReLU()),#H:64=>32 and W:128=>64
            Conv2dBlock(64,64,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),
            Conv2dBlock(64,64,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),
            
            Conv2dBlock(64,128,4,2,padding=padding_42,use_norm=True,use_act=True,act=nn.ReLU()),#16,32
            Conv2dBlock(128,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),
            Conv2dBlock(128,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),
            
            Conv2dBlock(128,256,4,2,padding=padding_42,use_norm=True,use_act=True,act=nn.ReLU()),#8,16
            Conv2dBlock(256,256,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),
            Conv2dBlock(256,256,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),
            
            Conv2dBlock(256,512,4,2,padding=padding_42,use_norm=True,use_act=True,act=nn.ReLU()),#4,8
            Conv2dBlock(512,512,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),
            Conv2dBlock(512,512,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),
        
            Conv2dBlock(512,512,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU()),#4,8
            Conv2dBlock(512,512,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU()),
            )        
        self.frames_fc = nn.Sequential(nn.Linear(4*8*512,512),
                                      nn.BatchNorm1d(512),
                                      nn.ReLU(),
                                      nn.Linear(512,256),
                                      nn.ReLU())

        
    def swap_channel_frame_axes(self,video):
        return video.permute(0,2,1,3,4)#B,F,C,H,W to B,C,F,H,W

    def forward(self, audio_frames, video_frames):
        """audio_frames is of shape : [batch_size,t,13] and video_frames is of shape : [batch_size,channels,height,width
        """
        if len(audio_frames.shape)==3:
            audio_frames = audio_frames.unsqueeze(1)#adding channel dim
        batch_size = audio_frames.shape[0]
        img_height = video_frames.shape[-2]
        video_frames = video_frames[...,img_height//2:,:] #consider only lower half of the image so new height is 256/2 = 128

        video_frames = self.swap_channel_frame_axes(video_frames)    
        video_embeddings = self.frames_3d_to_2d(video_frames)#out has no Temporal(i,e where frames are stacked) dimension | 3d (F,H,W) to 2d(H,W)
        video_embeddings = torch.squeeze(video_embeddings,dim=2)#B,C,F=1,H,W ==> B,C,H,W
        video_embeddings = self.frames_encoder(video_embeddings).reshape(batch_size, -1)
        video_embeddings = self.frames_fc(video_embeddings)
        
        # audio_frames = audio_frames.reshape(batch_size, 1, -1)
        audio_embeddings = self.audio_encoder(audio_frames).reshape(batch_size,-1)
        audio_embeddings = self.audio_fc(audio_embeddings)
        #see syncnet forward pass https://github.com/Rudrabha/Wav2Lip/blob/master/models/syncnet.py#L62-L63
        audio_embeddings = nn.functional.normalize(audio_embeddings,p=2,dim=1) 
        video_embeddings = nn.functional.normalize(video_embeddings,p=2,dim=1) 
        
        cosine_dist = nn.functional.cosine_similarity(audio_embeddings,video_embeddings)
        cosine_dist = cosine_dist.reshape(batch_size,-1)
        return (cosine_dist,)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))
