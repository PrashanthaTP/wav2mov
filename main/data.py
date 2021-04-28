""" provides utils for datasets and dataloaders """

import torch
from torchvision import transforms as vtransforms
from tqdm import tqdm
from collections import namedtuple
from torch.utils.data import DataLoader,random_split

from wav2mov.core.data.datasets import AudioVideoDataset
from wav2mov.core.data.transforms import ResizeGrayscale

DataloadersPack = namedtuple('dataloaders',('train','val'))


TO_Grayscale = vtransforms.Grayscale(1)
def get_dataset(options,config,hparams):
    hparams = hparams['data']
    root_dir = config['train_test_dataset_dir']
    filenames_txt = config['filenames_txt']
    video_fps = hparams['video_fps']
    audio_sf = hparams["audio_sf"]
    img_size = hparams['img_size']
    target_img_shape = (hparams['img_channels'],img_size,img_size)
    transform = ResizeGrayscale(target_img_shape)
    dataset = AudioVideoDataset(root_dir=root_dir,
                                filenames_text_filepath=filenames_txt,
                                audio_sf=audio_sf,
                                video_fps=video_fps,
                                num_videos=options.num_videos,
                                transform=transform)
    return dataset

def get_dataloaders(options,config,params,shuffle=True,get_mean_std=True,collate_fn=None):
    hparams = params['data']
    batch_size = hparams['mini_batch_size']
    dataset = get_dataset(options,config,params)
    N = len(dataset)
    # print(f'total videos : {N}')
    train_sz = (N*9)//10
    test_sz = N-train_sz
    train_ds , test_ds = random_split(dataset,[train_sz,test_sz])
    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn,pin_memory=True)
    test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn)
    
    if not all(hparams.get(item,None) for item in ('mean','std')):
        mean,std = get_mean_and_std(train_dl,params['img_channels'],'video')
        hparams['mean'] = [value for value in mean.tolist() ]
        hparams['std'] = [value for value in std.tolist() ]
        params.update('data',hparams)
    return DataloadersPack(train_dl,test_dl)


def get_mean_and_std(dataloader,img_channels,attr):
    #mean = E(X)
    #variance = E(X**2)- E(X)**2
    #standard deviation = variance**0.5
    print('[DEBUG] Calculating mean and standard deviation for the dataset.Please wait...')
    channels_sum,channels_squared_sum,num_batches = 0,0,0
    # num_items = 0
    for sample in tqdm(dataloader,ascii=True,total=len(dataloader)):
        data = getattr(sample,attr)
        data = data/255 #of shape (N,F,H,W,C)
        if img_channels==1:
            data = data.permute(0,1,4,2,3)
            data = TO_Grayscale(data)
            data  = data.permute(0,1,3,4,2)
        # print(data.shape)
        channels_sum += torch.mean(data,dim=[0,1,2,3])
        #except for the channel dimension as we want mean and std 
        # for each channel
        channels_squared_sum += torch.mean(data**2,dim=[0,1,2,3])
        # num_items += data.shape[0]
        num_batches += 1
    mean = channels_sum/num_batches     
    # print(channels_squared_sum,mean,channels_squared_sum/num_batches)
    std = ((channels_squared_sum/num_batches) - mean**2)**0.5
    # print(mean,std)
    return mean,std