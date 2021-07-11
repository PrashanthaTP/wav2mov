import os
import torch
from inference.models import Generator
from inference import params
from inference.utils import get_module_level_logger
logger = get_module_level_logger(__name__)

gen_hparams = {
    "in_channels": 3,
    "chs": [64, 128, 256, 512, 1024],
    "latent_dim": 272,
    "latent_dim_id": [8, 8],
    "comment": "laten_dim not eq latent_dim_id + latent_dim_audio, its 4x4 + 256",
    "latent_dim_audio": 256,
    "device": "cpu",
    "lr": 2e-4
}


def get_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = params.GEN_CHECKPOINT_PATH

    if not os.path.isfile(checkpoint_path):
        logger.error(f'NO FILE : {checkpoint_path}')
        raise FileNotFoundError(
            'Please make sure to put generator file in pt_files folder ')
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    model = Generator(gen_hparams)
    model.load_state_dict(state_dict)
    model.eval()
    return model
