"""
Main file of the Wav2Mov Project

It is the entry point to various functions


"""
import torch 
from torch.utils.tensorboard.writer import SummaryWriter 

from wav2mov.logger import Logger 
from wav2mov.config import config 
from wav2mov.params import params

from wav2mov.main.preprocess import create_from_grid_dataset
from wav2mov.main.train import train_model
from wav2mov.main.test import test_model

from wav2mov.main.options import Options

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
    
if __name__ == '__main__':
    options = Options().parse()
    
    if options.log in ['y','yes']:
        logger = get_logger(filehandler_required=True)
    else:
        logger = get_logger(filehandler_required=False)
        
    try:
        allowed = ('y','yes')
        if options.preprocess in allowed:
            preprocess(logger)
        
        if options.train in allowed:
            required = ('num_videos',)
      
            if not all(getattr(options,option) for option in required):
                raise RuntimeError('Cannot train without the options :',required)
            train(logger,options)
            
        if options.test in allowed:
            required = ('model_path',)
            
            if not all(getattr(options,option) for option in required):
                raise RuntimeError(f'Cannot test without the options : {required}')
            test(logger,options)
            
    except Exception as e: 
        logger.exception(e)
