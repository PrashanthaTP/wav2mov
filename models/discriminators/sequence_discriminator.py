
import torch
from torch import nn,optim
from torch.functional import Tensor
from torch.nn.modules.activation import LeakyReLU

from wav2mov.core.models.base_model import BaseModel
#GRU : Sequence of Frames
class SequenceDiscriminator(BaseModel):
    """
    >>>  self.gru = nn.GRU(input_size=in_size,hidden_size=h_size,num_layers=num_layers,batch_first = True)
    >>> out,_ = self.gru(x)#out is of shape (batch_size,seq_len,num_dir*hidden_dim)
    >>> return out[:,-1,:]
    """
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams 
        in_size, h_size, num_layers = self.hparams['in_size'],self.hparams['h_size'],self.hparams['num_layers']
        self.gru = nn.GRU(input_size=in_size,hidden_size=h_size,num_layers=num_layers,batch_first = True)
      
    def forward(self,x1,x2):
        # x1 and x2 are of shape (batch_size,in_channels.img_size,img_size)
        """ 
        For GRU  the input should be 3 dimensional 
        (batch_size,seq_len,num_features) #assuming batch_first is set to True while initializing GRU
        so flatten x1 and x2 which are 3 dimensional images and are thus 5 dimensional tensors after batching
        and catenate to get the shape (batch_size,2,num_channels*img_size*img_size)
        """
        batch_size = x1.shape[0]
        # print(x1.shape,self.hparams['in_channels'],x1.reshape(batch_size,1,-1).shape)
        x1 = torch.reshape(x1,(batch_size,self.hparams['in_channels'],-1))
        x2 = torch.reshape(x2,(batch_size,self.hparams['in_channels'],-1))
        # print(x1.shape,x2.shape)
        x1 = torch.cat([x1,x2],dim=1)
        # print(x1.shape)
        out,_ = self.gru(x1)#out is of shape (batch_size,seq_len,num_dir*hidden_dim)
        
        return out[:,-1,:]

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))



class SequenceDiscriminatorCNN(BaseModel):

    def __init__(self,hparams,use_bias=True):
        super().__init__()
        self.hparams = hparams
        self.in_channels = hparams['in_channels']
        self.chs = hparams['chs']
        chs = [self.in_channels] + self.chs
        self.cnn = nn.ModuleList([
            nn.Sequential(nn.Conv3d(chs[i], chs[i+1], (3,4,4), (1,2,2),(1,1,1),bias=use_bias),
                          nn.BatchNorm3d(chs[i+1]),
                          nn.LeakyReLU(0.2) ) for i in range(len(chs)-2)
            ])
        self.cnn.append(nn.Conv3d(chs[-2], chs[-1], (2, 4, 4), (2, 2, 2), (1, 1, 1)))
        
        """
        [8,64,128,256,1],
                    in          out
        filter 1 : 2x256x256  | ((2-2)+2)/2 +1 = 2
                                ((256-4+2)/2) + 1 = 
                                
                    20x256x256 | 20-3+2/1 + 1 = 20
        """ 
    def forward(self,*x):
        """sequence discriminator using 3d convolution for spatio temporal feature extraction
        """
      
        if len(x)==1 and isinstance(x[0],Tensor):
            x = x[0]
        else:
            x = torch.cat([frame.unsqueeze(dim=2) for frame in x if frame.dim()==4 ],dim=2)
            

        batch_size = x.shape[0]

        for cnn in self.cnn:
            x = cnn(x)
            # print(f'out shape : {x1.shape}')
        
        return x.reshape(batch_size,-1)
        

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))