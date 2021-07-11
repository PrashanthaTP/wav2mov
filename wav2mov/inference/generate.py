# get audio and image
# get audio_frames and extract face from image and resize
# apply transforms
# generator forward pass
import argparse
import os
import torch

from inference import audio_utils,image_utils,utils,model_utils
logger = utils.get_module_level_logger(__name__)

def is_exists(file):
    return os.path.isfile(file)

def create_batch(sample):
    return sample.unsqueeze(0)

def save_video(audio,video_frames):
    video_path = os.path.join(DIR,'out','fake_video_without_audio.avi')
    #####################################################################
    #⚠ IMPORTANT TO DE NORMALiZE ELSE OUTPUT WILL BE BLACK!
    #####################################################################
    video_frames = ((video_frames*0.5)+0.5)*255
    utils.save_video(video_path,audio,video_frames)
    
DIR = os.path.dirname(os.path.abspath(__file__))

def generate_video(image,audio):
    audio_feats = audio_utils.preprocess_audio(audio)
    num_frames = audio_feats.shape[0]
    image = image_utils.preprocess_image(image)
    images = image_utils.repeat_img(image,num_frames)
    logger.debug(f'✅ Preprocessing Done.')
    model = model_utils.get_model()
    logger.debug(f'✅ Model Loaded.')
    with torch.no_grad():
        model.eval()
        gen_frames = model(create_batch(audio_feats),create_batch(images))
        logger.debug(f'✅ Frames Generated.')
        B,T,*img_shape = gen_frames.shape
        gen_frames = gen_frames.reshape(B*T,*img_shape)
        save_video(audio,gen_frames)
    
if __name__ == '__main__':
    logger.debug(f'✅ Modules loaded.')
    # default_image = r'ref_frame_Run_27_5_2021__16_6.png'
    # default_image = r'ref_frame_Run_19_6_2021__12_14.png'
    # default_image = r'image.jfif'
    # default_image = r'musk.jfif'
    default_image = r'train_ref_frame_Run_9_7_2021__14_52.png'
    default_image = r'train_ref_frame_Run_11_7_2021__19_18.png'
    # default_audio = r'03-01-01-01-02-02-01.wav'
    default_audio = r'audio.wav'
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        description="Wav2Mov | Speech Controlled Facial Animation")
    arg_parser.add_argument('--image','-i',type=str,required=False,
                            help='image containing face of the person',
                            default=os.path.join(DIR,'inputs',default_image))
    arg_parser.add_argument('--audio','-a',type=str,required=False,
                            help='speech for which face should be animated',
                            default=os.path.join(DIR,'inputs',default_audio))
    options = arg_parser.parse_args()
    
    
    image_path = options.image
    audio_path = options.audio
    if not (is_exists(image_path)):
        raise FileNotFoundError(f'[ERROR] ❌ image path is incorrect :{image_path}')
    if not (is_exists(audio_path)):
        raise FileNotFoundError(f'[ERROR] ❌ audio path is incorrect :{audio_path}')
    image = image_utils.load_image(image_path)
    audio = audio_utils.load_audio(audio_path)
    generate_video(image,audio)
    logger.debug('✅ Video saved')
