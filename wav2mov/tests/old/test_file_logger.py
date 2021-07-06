from wav2mov.logger import Logger

def test():
    logger = Logger(__name__)
    logger.add_console_handler()
    logger.add_filehandler('logs/log.log')
    
    log = []
    for i in range(5):
        log.append(f'log {i}')
        
    logger.info('\t'.join(log))
    

if __name__ == '__main__':
    test()