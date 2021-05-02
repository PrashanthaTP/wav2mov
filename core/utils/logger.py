import logging
logging.basicconfig(level=logging.error,format="%(levelname)s : %(name)s : %(asctime)s | %(msg)s ")     

 
def get_module_level_logger(name):
    logger =  logging.getlogger(name)
    logger.setlevel(logging.debug)
    logger.propagate = False 
    return logger
 