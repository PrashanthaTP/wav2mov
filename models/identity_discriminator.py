import torch 
from torch import nn,optim
from torch.nn import functional as F

from wav2mov.core.models.base_model import BaseModel
from wav2mov.models.utils import init_net

class Block(nn.Module):
    def __init__(self,in_ch,out_ch,kernel,stride,padding,is_final_layer=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,in_ch*2,kernel,stride,padding)
        self.norm = nn.BatchNorm2d(in_ch*2)
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
        self.blocks = nn.ModuleList([Block(chs[i],chs[i+1],4,2,1) for i in range(len(chs)-2)])
        # blocks = [Block(chs[i],chs[i+1],4,2,1) for i in range(len(chs)-1)]
    
        self.desc = nn.Sequential(*self.blocks, Block(chs[-2], chs[-1], 4, 2, 1,is_final_layer=True))
        
    def forward(self,x,y):
        """
        x : frame image 
        y : still image
        """
        x = torch.cat([x,y],dim=1)
        # return self.desc(x)
        # for block in self.blocks:
        #     x = block(x)
        # return x.reshape(x.shape[0],-1)
        return self.desc(x)
    
    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))
   