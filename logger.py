import logging
import os
from torch.utils.tensorboard.writer import SummaryWriter
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
class TensorLogger:
    def __init__(self,runs_dir):
        self.writers = {}
        self.runs_dir = runs_dir
    def create_writer(self,name):
        self.writers[name] = SummaryWriter(os.path.join(self.runs_dir,name))
        
    def add_writer(self, name, writer):
        self.writers[name] = writer
        
    def add_writers(self,names):
        for name in names:
            self.create_writer(name)
            
    def add_scalar(self,writer_name,tag,scalar,global_step):
        if writer_name not in self.writers:
            logger.warning('No writer found named {writer_name}',writer_name=writer_name)
            self.create_writer(writer_name)
        self.writers[writer_name].add_scalar(tag,scalar,global_step)
    
    
    def add_scalars(self,d,global_step):
        for writer_name,(tag,scalar) in d.items():
            self.add_scalar(writer_name,tag,scalar,global_step)
    
    
    
class Logger:
    def __init__(self, name):
        self.log_fullpath = None
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
    def __add_handler(self, handler: logging.Handler):
        handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

    @classmethod
    def __get_formatter(cls, fmt: str):
        return logging.Formatter(fmt,"%b %d,%Y %H:%M:%S")

    def add_filehandler(self, log_fullpath, fmt:str=None):
        self.log_fullpath = log_fullpath
        os.makedirs(os.path.dirname(self.log_fullpath), exist_ok=True)
        if fmt is None:
            fmt = '[%(levelname)s] %(filename)s :  %(asctime)s : line no: %(lineno)d : %(message)s'
        file_handler = logging.FileHandler(log_fullpath)
        file_handler.setFormatter(self.__get_formatter(fmt))
        self.__add_handler(file_handler)

    def add_console_handler(self, fmt: str = None):
        if fmt is None:
            fmt = "[%(levelname)s] %(filename)s :  %(asctime)s : line no: %(lineno)d : %(message)s"
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.__get_formatter(fmt))
        self.__add_handler(console_handler)


   
    @property
    def debug(self):
        return self.logger.debug
    @property
    def info(self):
        return self.logger.info
    @property
    def warning(self):
        return self.logger.warning
    @property
    def error(self):
        return self.logger.error
    @property
    def exception(self):
        return self.logger.exception
    
    
"""
# def _get_logger_from_type(self,log_type_str):
#     if log_type_str == 'INFO':
#         return self.logger.info
#     if log_type_str == 'WARNING':
#         return self.logger.warning
#     if log_type_str == 'ERROR':
#         return self.logger.error
#     else:
#         return self.logger.debug
"""
# def log(self,message,log_type:str='DEBUG'):
#     """logs the message

#     Args:
#         message(str): message to be logged
#         log_type(str, optional): type of the log. Defaults to 'DEBUG'.
#     """
#     self._get_logger_from_type(log_type)(message)

