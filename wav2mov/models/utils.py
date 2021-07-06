from math import ceil
from torch.nn import init

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)

def get_same_padding(kernel_size,stride,in_size=0):
    """https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
       https://github.com/DinoMan/speech-driven-animation/blob/dc9fe4aa77b4e042177328ea29675c27e2f56cd4/sda/utils.py#L18-L21
    
    padding = 'same'
        • Padding such that feature map size has size In_size/Stride
        
        • Output size is mathematically convenient
        
        • Also called 'half' padding
        
    out = (in-k+2*p)/s + 1
    if out == in/s:
    in/s = (in-k+2*p)/s + 1 
    ((in/s)-1)*s + k -in = 2*p
    (in-s)+k-in = 2*p
    in case of s==1:
        p = (k-1)/2
    """
    out_size = ceil(in_size/stride)
    return ceil(((out_size-1)*stride+ kernel_size-in_size)/2)#(k-1)//2 for same padding

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    src : https://github1s.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py 
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    logger.debug(f'initializing {net.__class__.__name__} with {init_type} weights')
    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
 

    Return an initialized network.
    """
    
    init_weights(net, init_type, init_gain=init_gain)
    return net


def squeeze_batch_frames(target):
    batch_size,num_frames,*extra = target.shape
    return target.reshape(batch_size*num_frames,*extra)