"""
Main file of the Wav2Mov Project

It is the entry point to various functions


"""
import os
import shutil
import torch 
from torch.utils.tensorboard.writer import SummaryWriter 

from wav2mov.logger import Logger 
from wav2mov.config import get_config 

from wav2mov.params import params

from wav2mov.main.preprocess import create_from_grid_dataset
from wav2mov.main.train import train_model
from wav2mov.main.test import test_model

from wav2mov.main.options import Options,set_options
from wav2mov.main.validate_params import check_batchsize
torch.manual_seed(params['seed'])

def get_logger(config,filehandler_required=False):
    local_logger = Logger(__name__)
    local_logger.add_console_handler()
    if filehandler_required:
        local_logger.add_filehandler(config['log_fullpath'],in_json=True)
    return local_logger

def preprocess(preprocess_logger,config):
    create_from_grid_dataset(config,preprocess_logger)

def train(train_logger,args_options,config):
    train_model(args_options,params,config,train_logger)

def test(test_logger,args_options,config):
    test_model(args_options,params,config,test_logger)

def save_message(options,config):
    if not getattr(options,'msg'):return
    path = os.path.join(os.path.dirname(config['log_fullpath']),f'message_{config.version}.txt')
    print('message written to ',path)
    with open(path,'a+') as file:
        file.write(options.msg)
        file.write('\n')
        
def move_dir(src,dest):
    os.makedirs(dest, exist_ok=True)
    print('src :',src)
    print('dest :',dest)
    shutil.move(src, dest)
    print(f'[SHUTIL]: Folder moved : src ({src}) dest ({dest})')
    
def mov_log_dir(options,config):
    src = os.path.dirname(config['log_fullpath'])
    if os.path.exists(src):
        dest = os.path.join(os.path.join(config['base_dir'],'logs'),
                            options.version)
        # os.makedirs(dest,exist_ok=True)
        move_dir(src,dest)
    
def mov_out_dir(options,config):
    if not options.test in ['y', 'yes']:
        return
    run = os.path.basename(options.model_path).strip('gen_').split('.')[0]
    version_dir = os.path.dirname(os.path.dirname(options.model_path))
    print('version dir',version_dir)
    version = os.path.basename(version_dir)
    if 'v' not in version:
        version = options.version
    src = os.path.join(config['out_dir'], run)
    if os.path.exists(src):
        dest = os.path.join(config['out_dir'], version)
        # os.makedirs(dest,exist_ok=True)
        move_dir(src, dest)
        
def main(config):
    
    allowed = ('y', 'yes')
    if options.preprocess in allowed:
        preprocess(logger,config)

    if options.train in allowed:
        required = ('num_videos', 'num_epochs')

        if not all(getattr(options, option) for option in required):
            raise RuntimeError('Cannot train without the options :', required)
        check_batchsize(hparams=params['data'])
        train(logger, options,config)

    if options.test in allowed:
        required = ('model_path',)

        if not all(getattr(options, option) for option in required):
            raise RuntimeError(
                f'Cannot test without the options : {required}')
        test(logger, options,config)

if __name__ == '__main__':

    
    options = Options().parse()
    config = get_config(options.version)
    set_options(options,params)
    logger = get_logger(config,filehandler_required=options.log in ['y', 'yes'])
        
    save_message(options,config)
    try:
        main(config)
    except Exception as e: 
        if options.train in ['y','yes']:
            params.save(config['params_checkpoint_fullpath'])
        logger.exception(e)
        
    finally:
        logger.cleanup()
      
   
