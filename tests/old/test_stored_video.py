import torch 
from torchvision import transforms as vtransforms

import numpy as np
from torchvision.transforms.transforms import CenterCrop

import argparse 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gray','-g',choices=['y','n'],default='y',type=str,help='whether the output should be grayscale or color image')
args = parser.parse_args()
print('settings : gray(y/n): ',args.gray)
from wav2mov.utils.plots import show_img



path = r'E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\wav2mov\wav2mov\datasets\grid_dataset\s10_l_bbat9p\video_frames.npy'


def test_db():
    
    
    video = np.load(path).astype('float64')
    video = video.transpose(0,3,1,2)
    video = torch.from_numpy(video)
    print(f'video shape{video.shape}')
    
    
    
    
    channels = video.shape[1]
    print(' before '.center(10,'='))
    print('mean :',[torch.mean(video[0][:,i,...]) for i in range(channels)])
    print('std : ', [torch.std(video[0][:,i,...]) for i in range(channels)])
    video = video/255
    
    channels = 1 if args.gray == 'y' else 3
    transforms = vtransforms.Compose(

        [
            vtransforms.Grayscale(1),
            #  vtransforms.CenterCrop(256),
            vtransforms.Resize((256, 256)),
            vtransforms.Normalize([0.5]*channels, [0.5]*channels)
        ])
    print(' after '.center(10,'='))
    print('mean :',[torch.mean(video[0][:,i,...]) for i in range(channels)])
    print('std : ', [torch.std(video[0][:,i,...]) for i in range(channels)])
    print('max : ', [torch.max(video[0][:,i,...]) for i in range(channels)])
    print('min : ', [torch.min(video[0][:,i,...]) for i in range(channels)])
    video = transforms(video)
    # show_img(video[0])
    show_img(video[0],cmap='gray')
   

def main():
    test_db()
  
if __name__=='__main__':
    main()
