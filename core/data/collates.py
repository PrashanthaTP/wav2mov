""" collate functions """
from collections import namedtuple
import torch 
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
from wav2mov.core.data.utils import Sample,SampleWithFrames


from wav2mov.core.data.utils import AudioUtil
Lens = namedtuple('lens', ('audio', 'video'))




def collate_fn(batch):
    videos = [(torch.tensor(sample.video),(sample.video.shape[0])) for sample in batch]
    videos,video_lens = list(zip(*videos))
    audios = [(torch.tensor(sample.audio),(sample.audio.shape[0])) for sample in batch]
    audios,audio_lens = list(zip(*audios))
    videos = pad_sequence(videos, batch_first=True)
    audios = pad_sequence(audios, batch_first=True)
  
    return Sample(audios,videos),Lens(audio_lens,video_lens)


def get_frames_limit(audio_lens,video_lens,stride):
    audio_frames_lens = [audio_len//stride for audio_len in audio_lens]
    return min(min(audio_frames_lens),min(video_lens))

def video_frange_start(num_frames,req_frames):
    return (num_frames-req_frames)//2

def video_frange_end(num_frames,req_frames):
    return (num_frames + req_frames)//2

def get_batch_collate(hparams):
    stride =  hparams['audio_sf']// hparams['video_fps']
    audio_util = AudioUtil(hparams['coarticulation_factor'],stride)
    def collate_fn(batch):
        videos = [(torch.tensor(sample.video),sample.video.shape[0]) for sample in batch]
        videos,video_lens = list(zip(*videos))
        
        audios = [(torch.tensor(sample.audio).unsqueeze(0),sample.audio.shape[0]) for sample in batch]
        audios,audio_lens = list(zip(*audios))

        req_frames = get_frames_limit(audio_lens,video_lens,stride)
        ranges = [(video_frange_start(video.shape[0],req_frames),video_frange_end(video.shape[0],req_frames)) for video in videos]
        #[<------- total_frames-------->]        
        #[<---><---req_frames----><---->]
        #     ^                  ^
        #   frange_start        frange_end
        videos = [video[ranges[i][0]:ranges[i][1],... ].unsqueeze(0) for i,video in enumerate(videos)]
     
        audio_frames = [audio_util.get_audio_frames(audio,num_frames=req_frames).unsqueeze(0) for i,audio in enumerate(audios)]
        audios = [audio_util.get_limited_audio(audio,num_frames=req_frames) for audio in audios]

        return SampleWithFrames(torch.cat(audios),torch.cat(audio_frames),torch.cat(videos))
    
    return collate_fn