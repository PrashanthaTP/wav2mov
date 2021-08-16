import logging
logging.basicConfig(level=logging.DEBUG,format="%(levelname)s : %(name)s : %(asctime)s | %(msg)s ")     

 
def get_module_level_logger(name):
    logger =  logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False 
    return logger
 