""" provides utils for datasets and dataloaders """

import os
import numpy as np
import torch
from torchvision import transforms as vtransforms
from tqdm import tqdm
from collections import namedtuple
from torch.utils.data import DataLoader,random_split

from wav2mov.core.data.datasets import AudioVideoDataset
from wav2mov.core.data.transforms import ResizeGrayscale,Normalize

from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)

DataloadersPack = namedtuple('dataloaders',('train','val'))


To_Grayscale = vtransforms.Grayscale(1)

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

def get_dataloaders(options,config,params,shuffle=True,collate_fn=None):
    hparams = params['data']
    batch_size = hparams['mini_batch_size']
    dataset = get_dataset(options,config,params)
    N = len(dataset)
    # print(f'total videos : {N}')
    train_sz = (N*9)//10
    test_sz = N-train_sz
    train_ds , test_ds = random_split(dataset,[train_sz,test_sz])
    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn,pin_memory=True)
    test_dl = DataLoader(test_ds,batch_size=1,shuffle=shuffle,collate_fn=collate_fn)
    
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
    logger.debug('Calculating mean and standard deviation for the dataset.Please wait...')
    channels_sum,channels_squared_sum,num_batches = 0,0,0
    # num_items = 0
    for sample in tqdm(dataloader,ascii=True,total=len(dataloader)):
        data = getattr(sample,attr)
        data = data/255 #of shape (N,F,H,W,C)
        if img_channels==1:
            data = data.permute(0,1,4,2,3)#N,F,C,H,W
            data = To_Grayscale(data)
            data  = data.permute(0,1,3,4,2)#N,F,H,W,C
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

def get_dataset_v2(options,config,hparams):
    hparams = hparams['data']
    root_dir = config['train_test_dataset_dir']
    filenames_train_txt = config['filenames_train_txt']
    filenames_test_txt = config['filenames_test_txt']
    video_fps = hparams['video_fps']
    audio_sf = hparams["audio_sf"]
    img_size = hparams['img_size']
    target_img_shape = (hparams['img_channels'],img_size,img_size)

    num_videos_train = int(options.num_videos*0.9)
    num_videos_test = options.num_videos - num_videos_train
    
    mean_std_train = get_mean_and_std_v2(root_dir,filenames_train_txt,img_channels=1)
    to_gray_transform = ResizeGrayscale(target_img_shape)
    normalize = Normalize(mean_std_train)
    transforms_composed = vtransforms.Compose([to_gray_transform,normalize])
    dataset_train = AudioVideoDataset(root_dir=root_dir,
                                filenames_text_filepath=filenames_train_txt,
                                audio_sf=audio_sf,
                                video_fps=video_fps,
                                num_videos=num_videos_train,
                                transform=transforms_composed)
    dataset_test = AudioVideoDataset(root_dir=root_dir,
                                filenames_text_filepath=filenames_test_txt,
                                audio_sf=audio_sf,
                                video_fps=video_fps,
                                num_videos=num_videos_test,
                                transform=transforms_composed)
    return dataset_train,dataset_test

def get_dataloaders_v2(options,config,params,shuffle=True,collate_fn=None):
    hparams = params['data']
    batch_size = hparams['mini_batch_size']
    train_ds,test_ds = get_dataset_v2(options,config,params)
    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn,pin_memory=True)
    test_dl = DataLoader(test_ds,batch_size=1,shuffle=shuffle,collate_fn=collate_fn)
    return DataloadersPack(train_dl,test_dl)




def get_mean_and_std_v2(root_dir,filenames_txt,img_channels):
    logger.debug('Calculating mean and standard deviation for the dataset.Please wait...')
    ret = {}
    #mean = E(X)
    #variance = E(X**2)- E(X)**2
    #standard deviation = variance**0.5
    filenames_path = os.path.join(root_dir,filenames_txt)
    with open(filenames_path) as file:
        filenames = file.read().split('\n')
    for i,name in enumerate(filenames[:]):
        if not name.strip():
            del filenames[i]
            
    channels_sum,channels_squared_sum,num_batches = 0,0,0
    # num_items = 0
    for _,filename in tqdm(enumerate(filenames),ascii=True,total=len(filenames),desc='video'):
        video_path = os.path.join(root_dir,filename,'video_frames.npy')
        video = torch.from_numpy(np.load(video_path))
        video = video/255 #of shape (F,H,W,C)
        if img_channels==1:
            video = video.permute(0,3,1,2)#F,C,H,W
            video = To_Grayscale(video)
            video  = video.permute(0,2,3,1)
      
        channels_sum += torch.mean(video,dim=[0,1,2])
        #except for the channel dimension as we want mean and std 
        # for each channel
        channels_squared_sum += torch.mean(video**2,dim=[0,1,2])
        # num_items += video.shape[0]
        num_batches += 1
    mean = channels_sum/num_batches     
   
    std = ((channels_squared_sum/num_batches) - mean**2)**0.5
   
    ret['video']  = (mean,std)
    
    
    running_mean_sum , running_squarred_mean_sum =  0,0
    for _,filename in tqdm(enumerate(filenames),ascii=True,total=len(filenames),desc='audio'):
        audio_path = os.path.join(root_dir,filename,'audio.npy')
        audio = torch.from_numpy(np.load(audio_path))
        running_mean_sum += torch.mean(audio)
        running_squarred_mean_sum += torch.mean(audio**2)
    mean = running_mean_sum/len(filenames)
    std = ((running_squarred_mean_sum/len(filenames))-mean**2)**0.5
    ret['audio'] = (mean,std)
    logger.debug(f'[MEAN and STANDARD_DEVIATION] {ret}')
    return ret