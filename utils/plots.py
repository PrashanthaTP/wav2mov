from typing import List
import torch
from torch.functional import Tensor
import imageio
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

def no_grad_wrapper(fn):
    def wrapper(*args,**kwargs):
        with torch.no_grad():
            return fn(*args,**kwargs)
    return wrapper

@no_grad_wrapper
def show_img(img,cmap='viridis'):
    if isinstance(img,np.ndarray):
        img_np = img
    else:
        if len(img.shape)>3:
            img = img.squeeze(0)
        img_np = img.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    # print(img_np.shape)
    # print(img_np)
    if img_np.shape[2]==1:#if single channel
        img_np = img_np.squeeze(2)
    plt.imshow(img_np,cmap=cmap)
    plt.show()
    # 

def save_gif(gif_path,images,duration=0.5):    
    """creates gif 

    Args:
        gif_path (str): path where gif file should be saved
        images (torch.funcitonal.Tensor): tensor of images of shape (N,C,H,W)
    """
    if isinstance(images,Tensor):
        images = images.numpy()
  

    images = images.transpose(0,2,3,1).astype('uint8')
   
    imageio.mimsave(gif_path,images,duration=0.5)