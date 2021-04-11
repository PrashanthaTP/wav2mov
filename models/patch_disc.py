import torch
from torch import nn,optim

class PatchDiscriminator(nn.Module):
    """ 
    ref : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self,hparams,norm_layer=nn.InstanceNorm2d,use_bias = True):
        super().__init__()
        self.hparams = hparams
        input_nc = hparams['in_channels']
        ndf = hparams['ndf']
        n_layers = hparams['num_layers']
        use_bias = norm_layer==nn.InstanceNorm2d
        #batch normalization has affine = True so it has additional scaling and biasing/shifting terms by default.
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc*2,ndf,kw,2,padw),
                    nn.LeakyReLU(0.2,inplace=True)]#inplace=True saves memory 
        
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8) #multiplier is clipped at 8 : so if ndf is 64 max possible is 64*8=512
            sequence += [
                nn.Conv2d(ndf*nf_mult_prev,ndf*nf_mult,kw,2,padw,bias=use_bias),
                norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2,True)      
                ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers,8)
        sequence += [ 
                     nn.Conv2d(ndf*nf_mult_prev,ndf*nf_mult,kw,1,padw,bias=use_bias),
                     norm_layer(ndf*nf_mult),
                     nn.LeakyReLU(0.2,True)
                     ]
        
        sequence += [
                nn.Conv2d(ndf*nf_mult,1,kw,1,padw)
            ]
        
        self.disc = nn.Sequential(*sequence)
        
    def forward(self,frame_image,still_image):
        frame_image = torch.cat([frame_image,still_image],dim=1)#catenate images along the channel dimension
        #frame image now has a shape of (N,C+C,H,W)
        return self.disc(frame_image)
    
    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.5, 0.999))
