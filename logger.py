import logging
import os
import re
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

logging.basicConfig(level=logging.ERROR,format="%(levelname)-5s : %(name)s : %(asctime)s | %(msg)s ")
TIME_FORMAT = "%b %d,%Y %H:%M:%S"

from pythonjsonlogger import jsonlogger


 
def get_module_level_logger(name):
    m_logger =  logging.getLogger(name)
    m_logger.setLevel(logging.DEBUG)
    m_logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-5s : %(name)s : %(asctime)s | %(msg)s "))
    m_logger.addHandler(handler)
    return m_logger




class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self,log_record,record,message_dict):
        # print(log_record,vars(record),message_dict)
        # for key,value in vars(record).items():
        #     print(f'{key} : {value}')
        super().add_fields(log_record,record,message_dict)
        # if not log_record.get('timestamp'):
            # now = datetime.utcnow().strftime(TIME_FORMAT)
            # log_record['timestamp'] = now
        # if log_record.get('level'):
        #     log_record['level'] = log_record['level'].upper()
        # else:
        #     log_record['level'] = record.levelname
        if log_record.get('asctime'):
            log_record['asctime'] = datetime.utcnow().strftime(TIME_FORMAT)
     
 

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
            logger.warning(f'No writer found named {writer_name}')
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
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        self.is_json = False
        self.is_first_log = True
        
    def __add_handler(self, handler: logging.Handler):
        handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

    @classmethod
    def __get_formatter(cls, fmt: str):
        return logging.Formatter(fmt,TIME_FORMAT)
    
    @classmethod
    def __get_json_formatter(cls,fmt):
        return CustomJsonFormatter(fmt=fmt)
    
    def __json_log_begin(self):
        # print(f'inside json begin')
        with open(self.log_fullpath,'a+') as file:
            # print(file.read())
            file.write('[\n')
            # print(file.read())
            
    def __json_log_end(self):
        print('writing "]" to json log')
        with open(self.log_fullpath,'a+') as file:
            file.write(']\n')
   
            
    def add_filehandler(self, log_fullpath, fmt:str=None,in_json=False):
        self.log_fullpath = log_fullpath
        if fmt is None:
            fmt = '%(levelname)-5s : %(filename)s :  %(asctime)s : line no: %(lineno)d : %(message)s'
        if in_json:
            self.is_json = True
            folder = os.path.dirname(self.log_fullpath)
            filename= os.path.basename(self.log_fullpath)
            filename = filename.split('.')[0] + '.json'
           
            self.log_fullpath = os.path.join(folder,filename)
            
            self.__json_log_begin()
            file_handler = logging.FileHandler(self.log_fullpath)
            file_handler.setFormatter(self.__get_json_formatter(fmt))
        else:
            file_handler = logging.FileHandler(self.log_fullpath)
            file_handler.setFormatter(self.__get_formatter(fmt))
        # self.log_fullpath = re.sub(r'(\\)',os.sep,self.log_fullpath)
        os.makedirs(os.path.dirname(self.log_fullpath), exist_ok=True)
        self.__add_handler(file_handler)

    def add_console_handler(self, fmt: str = None):
        if fmt is None:
            fmt = "%(levelname)-5s : %(filename)s :  %(asctime)s : line no: %(lineno)d : %(message)s"
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.__get_formatter(fmt))
        self.__add_handler(console_handler)

    def __add_comma(self):
        if self.is_first_log:
            self.is_first_log = False
            return
        with open(self.log_fullpath,'a+') as file:
            file.write(',\n')
            
    @property
    def debug(self):
        if self.is_json:self.__add_comma()
        return self.logger.debug
    
    @property
    def info(self):
        if self.is_json:self.__add_comma()
        return self.logger.info
    
    @property
    def warning(self):
        if self.is_json:self.__add_comma()
        return self.logger.warning
    
    @property
    def error(self):
        if self.is_json:self.__add_comma()
        return self.logger.error
    
    @property
    def exception(self):
        if self.is_json:self.__add_comma()
        return self.logger.exception
    
    def cleanup(self):
        if self.is_json:
            self.__json_log_end()
    
