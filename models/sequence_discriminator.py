import torch
from torch import nn,optim
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

    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
        self.in_channels = hparams['in_channels']
        self.chs = hparams['chs']
        chs = [self.in_channels] + self.chs
        self.cnn = nn.ModuleList([
            nn.Sequential(nn.Conv3d(chs[i], chs[i+1], (2,4,4), (2,2,2),(1,1,1),bias=False),
                          nn.BatchNorm3d(chs[i+1]),
                          nn.LeakyReLU(0.2) ) for i in range(len(chs)-2)
            ])
        self.cnn.append(nn.Conv3d(chs[-2], chs[-1], (2, 4, 4), (2, 2, 2), (1, 1, 1)))
        
        
    def forward(self,x):
        """sequence discriminator using 3d convolution for spatio temporal feature extraction



        Args:
            x1 (Tensor): prev frame : shape = (N,num_channels,img_size)
            x2 (Tensor): curr frame : shape = (N,num_channels,img_size)
        """
        x1 = x[0]
        batch_size = x1.shape[0]
        for frame in x[1:]:
           x1 = torch.cat([x1,frame.unsqueeze(dim=2)])
      
        
        for cnn in self.cnn:
            x1 = cnn(x1)
            # print(f'out shape : {x1.shape}')
        
        return x1.reshape(batch_size,-1)
        

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5,0.999))


