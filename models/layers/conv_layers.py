import torch
from torch import nn

from wav2mov.models.utils import get_same_padding

class DoubleConv2dBlock(nn.Module):
    """height and width are halved in feature map"""
    def __init__(self, 
                 in_ch, 
                 out_ch,
                 use_norm=True,
                 act=None):
        super().__init__()
        self.use_norm = use_norm
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,padding=get_same_padding(3,1),bias=not self.use_norm)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2) if act is None else act()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        # print('double conv block',x.shape,type(x),x.device,next(self.parameters()).device)
        x = self.conv1(x)
        if self.use_norm :
          x = self.batch_norm(x) 
        return self.relu(self.conv2(self.relu(x)))

class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 padding,
                 use_norm=True,
                 use_act=True,act=None):
        
        super().__init__()
        self.use_norm = use_norm
        self.use_act = use_act
        self.act = nn.LeakyReLU(0.2) if act is None else act
        self.norm = nn.BatchNorm2d(out_ch)
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,bias=self.use_norm)
        
    def forward(self,x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x
    
class ConvTranspose2dBlock(nn.Module):
    def __init__(self,in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 padding,
                 use_norm=True,
                 use_act=True,
                 act=None):
        super().__init__()
        self.use_norm = use_norm
        self.use_act = use_act
        self.act = nn.LeakyReLU(0.2) if act is None else act
        self.norm = nn.BatchNorm2d(out_ch)
        self.conv = nn.ConvTranspose2d(in_ch,out_ch,kernel_size,stride,padding,bias=self.use_norm)
    
    def forward(self,x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x

class DoubleConvTranspose2d(nn.Module):
    def __init__(self,in_ch,
                 skip_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 padding,
                 use_norm=True,
                 use_act=True,
                 act=None):
        super().__init__()
        self.use_norm = use_norm
        self.use_act = use_act
        
        self.conv1 = nn.Conv2d(in_ch+skip_ch,in_ch,kernel_size=3,stride=1,padding=1,bias=False)
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.ConvTranspose2d(in_ch,out_ch,kernel_size,stride,padding)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2) if act is None else act
        
    def forward(self,x):
        x = nn.functional.relu(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        if self.use_norm:
            x = self.norm2(x)
        if self.use_act:
            x = self.act(x)
        return x