import torch
from torchvision import transforms as vtransforms

from wav2mov.logger import TensorLogger
from wav2mov.main.data import get_dataloaders
from wav2mov.utils.misc import AverageMetersList,ProgressMeter

def setup_tensor_logger(config):
    tensor_logger = TensorLogger(config['runs_dir'])
    writer_names = ['writer_gen', 'writer_sync_disc',
                    'writer_seq_disc', 'writer_id_disc']
    tensor_logger.add_writers(writer_names)
    return tensor_logger


def get_tensor_logger(options, config):
    return setup_tensor_logger(config) if options.log in ['y', 'yes'] else None


def add_to_board(tensor_logger, losses, global_step, scalar_type):
    for name, value in losses.items():
        writer_name = 'writer_' + name
        tensor_logger.add_scalar(writer_name, scalar_type+'_'+name, value, global_step)


def add_img_grid(tensor_logger, img_grid, global_step, img_type):
    tensor_logger.add_image(img_grid, img_type, global_step)


def get_train_dl(options,config, hparams):
    loaders, mean, std = get_dataloaders(options,config, hparams, shuffle=True)
    train_dl = loaders.train
    return train_dl, mean, std


def get_transforms(img_size, img_channels):
    transforms = vtransforms.Compose(
        [
            vtransforms.Grayscale(1),
            vtransforms.Resize(img_size),
            vtransforms.Normalize([0.5]*img_channels, [0.5]*img_channels)
        ])
    return transforms


def get_device(hparams):
    if hparams['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


def load_checkpoint(options):
    return torch.load(options.model_path)


def get_meters(hparams,num_batches):
    num_epochs = hparams['num_epochs']
    batch_loss_meters = AverageMetersList(('id_disc', 'sync_disc', 'seq_disc', 'gen'),
                                          fmt=':0.4f')  # per video
    epoch_loss_meters = AverageMetersList(('id_disc', 'sync_disc', 'seq_disc', 'gen'),
                                          fmt=':0.4f')  # per epoch

    epoch_progress_meter = ProgressMeter(num_epochs, epoch_loss_meters.as_list(),prefix='[EPOCH]')
    batch_progress_meter = ProgressMeter(num_batches, batch_loss_meters.as_list(),prefix='[BATCH]')

    return batch_loss_meters, epoch_loss_meters, epoch_progress_meter,batch_progress_meter

def get_meters_v2(hparams,num_batches):
    num_epochs = hparams['num_epochs']
    batch_loss_meters = AverageMetersList(('id_disc', 'sync_disc', 'seq_disc', 'gen','l1'),
                                          fmt=':0.4f')  # per video
    epoch_loss_meters = AverageMetersList(('id_disc', 'sync_disc', 'seq_disc', 'gen','l1'),
                                          fmt=':0.4f')  # per epoch

    epoch_progress_meter = ProgressMeter(num_epochs, epoch_loss_meters.as_list(),prefix='[EPOCH]')
    batch_progress_meter = ProgressMeter(num_batches, batch_loss_meters.as_list(),prefix='[BATCH]')

    return batch_loss_meters, epoch_loss_meters, epoch_progress_meter,batch_progress_meter
