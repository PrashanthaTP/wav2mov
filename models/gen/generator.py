""" Generator based on UNET architecture """
from PIL import Image
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as vtransforms

from wav2mov.core.models.base_model import BaseModel
from wav2mov.models.utils import init_net,squeeze_batch_frames

class DoubleConvBlock(nn.Module):
    """
    Two convolution layers one followed by another
    Uses kernel size of 3 and stride of 1 and padding 0
    each convolution operation is followed by a relu activation

    After each convolution operation the new height and width reduce by two units
    + H_out = H_in - 2 
    +  W_out = W_in - 2

    Example:
            >>> enc_block = DoubleConvBlock(1, 64)
            >>> x  = torch.randn(1, 1, 572, 572)
            >>> print(enc_block(x).shape)
            >>> torch.Size([1, 64, 568, 568])


    """

    def __init__(self, in_ch, out_ch,use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,bias=not self.use_norm)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        # print('double conv block',x.shape,type(x),x.device,next(self.parameters()).device)
        x = self.conv1(x)
        if self.use_norm :
          x = self.batch_norm(x) 
        return self.relu(self.conv2(self.relu(x)))

class Encoder(nn.Module):
    """
    Contracting path of the UNET 
    It extracts meaningful feature map from an input image.As is standard practice 
    for a CNN , the Encoder,doubles the number of channels at everystep and halves 
    the spatial dimenstion 

    From the paper :
            "The contractive path consists of the repeated application of 
            two 3x3 convolutions (unpadded convolutions),
            each followed by a rectified linear unit (ReLU) and
            a 2x2 max pooling operation with stride 2 for downsampling.
            At each downsampling step we double the number of feature channels."

    Example:

            >>> encoder = Encoder()
            >>> # input image
            >>> x    = torch.randn(1, 3, 572, 572)
            >>> ftrs = encoder(x)
            >>> for ftr in ftrs: print(ftr.shape)
            >>> torch.Size([1, 64, 568, 568])
            >>> torch.Size([1, 128, 280, 280])
            >>> torch.Size([1, 256, 136, 136])
            >>> torch.Size([1, 512, 64, 64])
            >>> torch.Size([1, 1024, 28, 28])
            
            for input image of shape 256,256
            torch.Size([1, 64, 252, 252])
            torch.Size([1, 128, 122, 122])
            torch.Size([1, 256, 57, 57])  
            torch.Size([1, 512, 24, 24])  
            torch.Size([1, 1024, 8, 8])   
    """

    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [DoubleConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        filters = []
        for block in self.enc_blocks:
            x = block(x)
            filters.append(x)
            x = self.pool(x)
        return filters


class Decoder(nn.Module):
    """
    The expansive path of the UNET 
    From the paper:
            Every step in the expansive path consists of an upsampling 
            of the feature map followed by a 2x2 convolution (“up-convolution”) 
            that halves the number of feature channels, a concatenation with 
            the correspondingly cropped feature map from the contracting path, 
            and two 3x3 convolutions, each followed by a ReLU. 
            The cropping is necessary due to the loss of border pixels 
            in every convolution.

    Example:
            >>> decoder = Decoder()
            >>> x = torch.randn(1, 1024, 28, 28)
            >>> decoder(x, encoder_out[::-1][1:]).shape

            >>> (torch.Size([1, 64, 388, 388])

    """

    def __init__(self, up_chs=(1026, 512, 256, 128, 64),dec_chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.up_chs = up_chs
        self.dec_chs = dec_chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(self.up_chs[i], self.up_chs[i+1], 2, 2)
                                      for i in range(len(self.up_chs)-1)])
        self.dec_blocks = nn.ModuleList([DoubleConvBlock(self.dec_chs[i], self.dec_chs[i+1])
                                         for i in range(len(self.dec_chs)-1)])

    def forward(self, x, encoded_features):
        """
        >>> up torch.Size([1, 512, 56, 56])=>[(H-1)/s] + k -2*p = [(64-1)/2]+2-2*0 = 
        >>> enc torch.Size([1, 512, 64, 64])=>croppped to 56x56
        >>> cat torch.Size([1, 1024, 56, 56]) #cat(up,conv)
        """
        for i in range(len(self.up_chs)-1):
            #len-1 because conv(in,out) are in pairs
            # so if there are 4 chs then there are 3 conv blocks
            x = self.upconvs[i](x)
            # print('inside decoder unet ',x.shape,encoded_features[i].shape)
            # torch.Size([1, 512, 16, 16]) torch.Size([1, 512, 24, 24])
            cropped_enc_feat = self.crop(encoded_features[i], x)
            # print('up',x.shape)
            x = torch.cat([x, cropped_enc_feat], dim=1)
            # print('enc',encoded_features[i].shape)
            # print('cat',x.shape)

            x = self.dec_blocks[i](x)
        return x

    def crop(self, encoded_features, x):
        _, _, H, W = x.shape
        try:
            return vtransforms.Resize((H, W),interpolation= vtransforms.InterpolationMode.BICUBIC)(encoded_features)
        except:
            return vtransforms.Resize((H,W),interpolation=Image.BICUBIC)(encoded_features)

class Generator(BaseModel):
    """    Unet Architecture    https://arxiv.org/abs/1505.04597

            Kernel 4x4,stride 2, padding 1
            Leaky Relu in Encoder and ReLU in Decoder(Tanh at the output)

            >>> enc_filters = self.encoder(x)
            >>> out = self.decoder(enc_filters[::-1][0],enc_filters[::-1][1:])
            >>> out = self.head(out)
            >>> if self.retain_dim:
            >>>         out = F.interpolate(out,self.img_dim)
            >>> return out

    """

    def __init__(self, hparams):

        super().__init__()
        self.hparams = hparams

        self.img_dim = self.hparams['img_dim']
        self.retain_dim = self.hparams['retain_dim']

        enc_chs = [self.hparams['in_channels']] + self.hparams['enc_chs']
        dec_chs = self.hparams['dec_chs']
        up_chs = self.hparams['up_chs']
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(up_chs=up_chs,dec_chs=dec_chs)
        
        init_net(self.encoder)
        init_net(self.decoder)
        
        self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], self.hparams['in_channels'], 1),
                                  nn.Tanh())  # ((388-3)/1)+1 = 385

    def forward(self, frame_img, audio_noise):
        frame_img = squeeze_batch_frames(frame_img)#self.encoder is full of 3d cnn,but frame imgs is video having frame_num as second dim
        enc_filters = self.encoder(frame_img)
        # channel wise catenation
        # print(audio_noise.shape,enc_filters[::-1][0].shape)
        enc_filters = enc_filters[::-1]
        # print(f'inside gen {enc_filters[0].shape} {audio_noise.shape} {frame_img.shape}')
        enc_filters[0] = torch.cat([enc_filters[0], audio_noise], dim=1)
        # print(enc_filters[0].shape)
        out = self.decoder(enc_filters[0], enc_filters[1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.img_dim)
        return out


class AudioEnocoder(nn.Module):
    """ 
       >>> x = x.reshape(x.shape[0], 1, -1)
       >>> x = self.conv(x)
       >>> x = self.fc(x.reshape(x.shape[0],-1))
       >>> return x #shape (batch_size,28*28)
    """

    def __init__(self):
        super().__init__()
        use_bias = True
        #audio = 666*5=3330
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 330, 30,bias=use_bias),#input 666 output (3330-330+0)/30 + 1 = 101
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, 1,bias=use_bias),#((101-3+0)/1)+1 =98
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.BatchNorm1d(128, 1,4,2,1),#((98-4+2)/2)+1 = 48 +1 = 49
            nn.ReLU(),
        
        
            )
        # self.audio_fc = nn.Sequential(nn.Linear(49*256,5*256),
        #                               nn.ReLU(),
        #                               nn.Linear(5*256,64),
        #                               nn.ReLU())
        self.features_len = 49#out of conv layers
        self.num_layers = 1
        self.hidden_size = 8*8
        self.gru = nn.GRU(input_size=self.features_len,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True)            
    def forward(self, x):
        # print(x.shape) 
        batch_size,num_frames,features = x.shape
        x = self.conv(squeeze_batch_frames(x).unsqueeze(1))
        # print(x.shape) #=> (1,1,74)
        # x = self.audio_fc(x.reshape(x.shape[0], -1))
        x.reshape(batch_size,num_frames,self.features_len)
        x,_ = self.gru(x)
        #B,seq_len,hidden_size
        return x  # shape (batch_size,8*8)


class NoiseGenerator(nn.Module):
    """
        >>> noise = torch.randn(1,100)
        >>> return self.fc(noise)#shape (batch_size,28*28) 
    """

    def __init__(self,hparams):
        super().__init__()
        self.device = hparams['device']
        # self.fc = nn.Sequential(
        #     nn.Linear(100, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),#out dim is based on encoder part bottlenck of generator
        #     nn.ReLU()
        #     # nn.Linear(64, 28*28),
        #     # nn.ReLU()
        # )
        self.features_len = 10
        self.hidden_size = 8*8#see catenation of encoded noise and audio
        self.gru = nn.GRU(input_size=self.features_len,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True)
        #input should be of shape batch_size,seq_len,input_size
    def forward(self,batch_size,num_frames):
        noise = torch.randn(batch_size,num_frames,self.features_len,device=self.device)
        out,_ = self.gru(noise)
        return out#(batch_size,seq_len,hidden_size)


class GeneratorBW(BaseModel):
    """ 
        >>> x = torch.cat([self.audio_enc(audio).reshape(-1,1,28,28),self.noise_enc().reshape(-1,1,28,28)],dim=1)
        >>> return self.identity_enc(frame_img,x)
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.audio_enc = AudioEnocoder()
        self.noise_enc = NoiseGenerator(self.hparams)
        self.identity_enc = Generator(self.hparams)

        init_net(self.audio_enc)
        init_net(self.noise_enc)
        init_net(self.identity_enc)
        

    def forward(self, audio_frames, ref_video_frames):
        #audio : B,F,Sw
        #ref : B,F,C,H,W
        batch_size,num_frames,_ = audio_frames.shape
        _,_,*img_shape = ref_video_frames.shape
        # audio_frames = squeeze_batch_frames(audio_frames)
        # ref_video_frames = squeeze_batch_frames(ref_video_frames)

        x = torch.cat([self.audio_enc(audio_frames).reshape(-1, 1, 8, 8),
                       self.noise_enc(batch_size,num_frames).reshape(-1, 1, 8, 8)], dim=1)#channel wise
        x = self.identity_enc(ref_video_frames, x)
        return x.reshape(batch_size,num_frames,*img_shape)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5, 0.999))
