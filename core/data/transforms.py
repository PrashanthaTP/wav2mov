from torchvision import transforms as vtransforms

from wav2mov.core.data.utils import Sample
class ResizeGrayscale:
    def __init__(self,target_shape):
        img_channels ,*img_size = target_shape
        self.transform =  vtransforms.Compose(
                [
                    vtransforms.Grayscale(1),
                    vtransforms.Resize(img_size),
                    vtransforms.Normalize([0.5]*img_channels, [0.5]*img_channels)
                ])
    def __call__(self,sample):
        video = sample.video
        # print(f'inside transform ',type(video))
        video = self.transform(video)
        # print('after transform',type(video))
        return Sample(sample.audio,video) 
    