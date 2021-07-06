from wav2mov.logger import Logger 

log_path = r'E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\wav2mov\wav2mov\tests\log.json'
def test_logger():
    logger = Logger(__file__)
    logger.add_console_handler()
    logger.add_filehandler(log_path,in_json=True)
    logger.debug('testing json logger %d',extra={'great':1})
    logger.debug('testing json logger',extra={'great':1})
    logger.debug('testing json logger',extra={'great':1})
    logger.debug('testing json logger',extra={'great':1})

def main():
    test_logger()
if __name__=='__main__':
    main()
