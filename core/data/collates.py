""" collate functions """
from collections import namedtuple
import torch 
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
from wav2mov.core.data.utils import Sample

Lens = namedtuple('lens', ('audio', 'video'))




def collate_fn(batch):
    videos = [(torch.tensor(sample.video),(sample.video.shape[0])) for sample in batch]
    videos,video_lens = list(zip(*videos))
    audios = [(torch.tensor(sample.audio),(sample.audio.shape[0])) for sample in batch]
    audios,audio_lens = list(zip(*audios))
    videos = pad_sequence(videos, batch_first=True)
    audios = pad_sequence(audios, batch_first=True)
  
    return Sample(audios,videos),Lens(audio_lens,video_lens)
