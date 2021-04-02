"""
Main file of the Wav2Mov Project

It is the entry point to various functions


"""
import os
import torch 
from torch.utils.tensorboard.writer import SummaryWriter 

from wav2mov.logger import Logger 
from wav2mov.config import config 
from wav2mov.params import params

from wav2mov.main.preprocess import create_from_grid_dataset
from wav2mov.main.train_v3 import train_model
from wav2mov.main.test import test_model

from wav2mov.main.options import Options,set_options
from wav2mov.main.validate_params import check_batchsize
torch.manual_seed(params['seed'])

def get_logger(filehandler_required=False):
    local_logger = Logger(__name__)
    local_logger.add_console_handler()
    if filehandler_required:
        local_logger.add_filehandler(config['log_fullpath'])
    return local_logger


    
def preprocess(preprocess_logger):
    create_from_grid_dataset(config,preprocess_logger)

def train(train_logger,args_options):
    train_model(args_options,params,config,train_logger)

def test(test_logger,args_options):
    test_model(args_options,params,config,test_logger)


def save_message(options):
    if not getattr(options,'msg'):return
    path = os.path.join(os.path.dirname(config['log_fullpath']),f'message_{config.version}.txt')
    print('message written to ',path)
    with open(path,'a+') as file:
        file.write(options.msg)
        file.write('\n')
        

if __name__ == '__main__':
    options = Options().parse()
    set_options(options,params)
    if options.log in ['y','yes']:
        logger = get_logger(filehandler_required=True)
    else:
        logger = get_logger(filehandler_required=False)
    save_message(options)
    try:
        allowed = ('y','yes')
        if options.preprocess in allowed:
            preprocess(logger)
        
        if options.train in allowed:
            required = ('num_videos','num_epochs')
      
            if not all(getattr(options,option) for option in required):
                raise RuntimeError('Cannot train without the options :',required)
            check_batchsize(hparams=params['data'])
            train(logger,options)
            
        if options.test in allowed:
            required = ('model_path',)
            
            if not all(getattr(options,option) for option in required):
                raise RuntimeError(f'Cannot test without the options : {required}')
            test(logger,options)
            
    except Exception as e: 
        params.save(config['params_checkpoint_fullpath'])
        logger.exception(e)
