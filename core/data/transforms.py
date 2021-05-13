from torchvision import transforms as vtransforms

from wav2mov.core.data.utils import Sample

class ResizeGrayscale:
    def __init__(self,target_shape):
        _,*img_size = target_shape
        self.transform =  vtransforms.Compose(
                [
                    vtransforms.Grayscale(1),
                    vtransforms.Resize(img_size),
                    # vtransforms.Normalize([0.5]*img_channels, [0.5]*img_channels)
                ])
    def __call__(self,sample):
        video = sample.video
        video = self.transform(video)
        return Sample(sample.audio,video) 
    
class Normalize:
    def __init__(self,mean_std):
        audio_mean,audio_std = mean_std['audio'] 
        video_mean,video_std = mean_std['video']
        
        self.audio_transform = lambda x: (x-audio_mean)/audio_std
        self.video_transform = vtransforms.Normalize(video_mean,video_std)   
        
    def __call__(self,sample):
        audio,video = sample.audio,sample.video
        audio = self.audio_transform(audio)
        video = self.video_transform(video)
        return Sample(audio,video)