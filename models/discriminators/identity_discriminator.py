import torch 
from torch import nn,optim
from torch.nn import functional as F

from wav2mov.core.models.base_model import BaseModel
from wav2mov.models.utils import init_net,squeeze_batch_frames

class Block(nn.Module):
    def __init__(self,in_ch,out_ch,kernel,stride,padding,is_final_layer=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,in_ch*2,kernel,stride,padding)
        # self.norm = nn.BatchNorm2d(in_ch*2)
        self.conv2 = nn.Conv2d(in_ch*2,out_ch,kernel,stride,padding)
        self.is_final_layer = is_final_layer
    def forward(self,x):
        if self.is_final_layer:
            return self.conv2(F.relu(self.norm(self.conv1(x))))

        return F.relu(self.conv2(F.relu(self.norm(self.conv1(x)))))
    

#Current Frame Image and Still Image
class IdentityDiscriminator(BaseModel):
    """[summary]
        >>> super().__init__()
        >>> chs = (6,16,32,1)
        >>> self.blocks = nn.ModuleList([Block(chs[i],chs[i+1],4,2,1) for i in range(len(chs)-1)])
        >>> # blocks = [Block(chs[i],chs[i+1],4,2,1) for i in range(len(chs)-1)]
        >>> self.desc = nn.Sequential(*self.blocks)

        >>> x = torch.cat([x,y],dim=1)
        >>> # return self.desc(x)
        >>> for block in self.blocks:
        >>>     x = block(x)
        >>> return x.reshape(x.shape[0],-1)
    
    """
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
        chs = [self.hparams['in_channels']*2]+self.hparams['chs']
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv2d(chs[i],chs[i+1],4,2,1),nn.ReLU()) for i in range(len(chs)-2))
        # blocks = [Block(chs[i],chs[i+1],4,2,1) for i in range(len(chs)-1)]
    
        self.disc = nn.Sequential(*self.blocks, nn.Conv2d(chs[-2], chs[-1], 4, 2, 1))
        
    # def _squeeze_batch_frames(self,target):
    #     batch_size,num_frames,*extra = target.shape
    #     return target.reshape(batch_size*num_frames,*extra)

    def forward(self,x,y):
        """
        x : frame image (B,F,H,W)
        y : still image
        """
        assert x.shape==y.shape
        is_frame_dim_present = False

        if len(x.shape)>4:#frame dim present
            is_frame_dim_present = True
            batch_size,num_frames,*img_shape = x.shape
            x = squeeze_batch_frames(x)
            y = squeeze_batch_frames(y)
        
            
        x = torch.cat([x,y],dim=1)#along channels
        # return self.desc(x)
        # for block in self.blocks:
        #     x = block(x)
        # return x.reshape(x.shape[0],-1)
        x = self.disc(x)
        return x
        # return x if  not is_frame_dim_present else x.reshape(batch_size,num_frames,*img_shape)
    
    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))



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


class IdentityDiscriminator_V2(nn.Module):
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

    def __init__(self,hparams ,chs=(2, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [DoubleConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, frame_img,ref_image):
      
        filters = []
        for block in self.enc_blocks:
            x = block(x)
            filters.append(x)
            x = self.pool(x)
        return filters


   