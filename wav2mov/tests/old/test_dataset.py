import math
import os
import torch 
from wav2mov.core.data.datasets import AudioVideoDataset
from wav2mov.utils.audio import StridedAudio
from settings import BASE_DIR
ROOT_DIR = os.path.join(BASE_DIR,'datasets','grid_dataset')

def test():
    strided_audio = StridedAudio(16000//24,1)
    dataset = AudioVideoDataset(ROOT_DIR,os.path.join(ROOT_DIR,'filenames.txt'),video_fps=24,audio_sf=16000)
    for i,sample in enumerate(dataset):
        audio,video = sample
        stride = math.floor(16_000/24)
        print(f"Stride = {stride}")
        print(f"Audio shape,video shape , audio.shape[0]//stride")
        print(audio.shape,video.shape,audio.shape[0]//stride)
        get_frames = strided_audio.get_frame_wrapper(audio)
        for i in range(video.shape[0]):
            frame = get_frames(i)
            print(frame[0].shape,frame[1])
        break
        print("="*10)
        if i==3:
            break 
def main():
    test()
    return
if __name__=='__main__':
    main()