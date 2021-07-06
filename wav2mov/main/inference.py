# get audio and image
# get audio_frames and extract face from image and resize
# apply transforms
# generator forward pass

import argparse
import cv2
import dlib
import librosa
import os
import torch
from torchvision import utils as vutils
from torchvision import transforms as vtransforms

import logging 
def get_module_level_logger(name):
    m_logger =  logging.getLogger(name)
    m_logger.setLevel(logging.DEBUG)
    m_logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-5s : %(name)s : %(asctime)s | %(msg)s ", "%b %d,%Y %H:%M:%S"))
    m_logger.addHandler(handler)
    return m_logger
logger = get_module_level_logger(__name__)

# import Generator
# import hparams
def is_exists(file):
    return os.path.isfile(file)
def get_model():
    pass
def load_audio(audio_path):
    pass
def load_image(image_path):
    pass

def preprocess_image(image):
    pass
def preprocess_audio(audio):
    pass


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        description="Wav2Mov | Speech Controlled Facial Animation")
    arg_parser.add_argument('--image','-i',type=str,required=True,help='image containing face of the person')
    arg_parser.add_argument('--audio','-a',type=str,required=True,help='speech for which face should be animated')
    options = arg_parser.parse_args()
