import torch
from torch import nn

from wav2mov.models.sequence_discriminator import SequenceDiscriminatorCNN


def get_shape(in_shape,kernels,strides,padding):
    
    new_shape = []
    for i,k,s,p in zip(in_shape,kernels,strides,padding):
        new_shape.append(((i-k+2*p)/s)+1)
        
    return new_shape
        
def test(hparams):
    x1 = torch.randn(1,1,256,256) #N,Cin,Depth,Height,Width
    x2 = torch.randn(1,1,256,256)
    model = SequenceDiscriminatorCNN(hparams)
    
    print(model(x1,x2).shape)
    strides = (2,2,2)
    padding = (1,1,1)
    in_shape = (2,256,256)
    kernels = (2,4,4)
    print(get_shape(in_shape,kernels,strides,padding))
    """
    out shape : torch.Size([1, 6, 2, 128, 128])
    out shape : torch.Size([1, 32, 2, 64, 64])
    out shape : torch.Size([1, 64, 2, 32, 32])
    out shape : torch.Size([1, 32, 2, 16, 16])
    out shape : torch.Size([1, 16, 2, 8, 8])
    out shape : torch.Size([1, 1, 2, 4, 4])
    torch.Size([1, 32]) 
    
    1=>4=>8=>16=>8=>4=>1
    out shape : torch.Size([1, 4, 2, 128, 128])
    out shape : torch.Size([1, 8, 2, 64, 64])
    out shape : torch.Size([1, 16, 2, 32, 32])
    out shape : torch.Size([1, 8, 2, 16, 16])
    out shape : torch.Size([1, 4, 2, 8, 8])
    out shape : torch.Size([1, 1, 2, 4, 4])
    torch.Size([1, 32])
    [2.0, 128.0, 128.0]
    """
    
def main():
    hparams ={'in_channels':1,'chs':[4,8,16,8,4,1]}
    test(hparams)
    
if __name__=='__main__':
    main()
