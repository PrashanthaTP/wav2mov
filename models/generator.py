""" Generator based on UNET architecture """
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as vtransforms

from wav2mov.core.models.base_model import BaseModel
from wav2mov.models.utils import init_net

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

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        # print('double conv block',x.shape,type(x),x.device,next(self.parameters()).device)
        return self.relu(self.conv2(self.relu(self.conv1(x))))


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
            x = self.upconvs[i](x)
            cropped_enc_feat = self.crop(encoded_features[i], x)
            # print('up',x.shape)
            x = torch.cat([x, cropped_enc_feat], dim=1)
            # print('enc',encoded_features[i].shape)
            # print('cat',x.shape)

            x = self.dec_blocks[i](x)
        return x

    def crop(self, encoded_features, x):
        _, _, H, W = x.shape
        return vtransforms.CenterCrop([H, W])(encoded_features)


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
        
        self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], self.hparams['in_channels'], 1),nn.Tanh())  # ((388-3)/1)+1 = 385

    def forward(self, frame_img, audio_noise):
        enc_filters = self.encoder(frame_img)
        # channel wise catenation
        # print(audio_noise.shape,enc_filters[::-1][0].shape)
        enc_filters = enc_filters[::-1]
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
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, 3, 3),  # ((666-3)/3)+1 = 663/3 +1 = 222
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Conv1d(1, 1, 3, 3),  # ((222-3)/3)+1 = 219/3+1 = 73+1 = 74
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(74, 128), nn.ReLU(),
                                nn.Linear(128, 64), nn.ReLU())

    def forward(self, x):
        # print(x.shape) 
        x = x.reshape(x.shape[0], 1, -1)
        x = self.conv(x)
        # print(x.shape) #=> (1,1,74)
        x = self.fc(x.reshape(x.shape[0], -1))
  
        return x  # shape (batch_size,28*28)


class NoiseGenerator(nn.Module):
    """
        >>> noise = torch.randn(1,100)
        >>> return self.fc(noise)#shape (batch_size,28*28) 
    """

    def __init__(self,hparams):
        super().__init__()
        self.device = hparams['device']
        self.fc = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 64),#out dim is based on encoder part bottlenck of generator
            nn.ReLU()
            # nn.Linear(64, 28*28),
            # nn.ReLU()
        )

    def forward(self):
        noise = torch.randn(1, 100).to(self.device)
        return self.fc(noise)  # shape (batch_size,28*28)


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
        
    def forward(self, audio, frame_img):
        # batch_size = audio.shape[0]
 
        x = torch.cat([self.audio_enc(audio).reshape(-1, 1, 8, 8),
                       self.noise_enc().reshape(-1, 1, 8, 8)], dim=1)
    
        return self.identity_enc(frame_img, x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5, 0.999))
