from wav2mov.main import callbacks

def test():
    m_logger = callbacks.m_logger
    m_logger.debug(f'{m_logger.name} {m_logger.level}')
if __name__ == '__main__':
    test()