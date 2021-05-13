import time
from torch.utils.tensorboard.writer import SummaryWriter
from wav2mov.logger import TensorLogger


def test():
    logger = TensorLogger('logs')
    writer_1 = SummaryWriter('logs/exp1/test1')
    writer_2 = SummaryWriter('logs/exp1/test2')
    logger.add_writer('test1',writer_1)
    logger.add_writer('test2',writer_2)
    
    for i in range(10):
        logger.add_scalar('test1','aytala',i+1,i)
        print('adding 1 ',i)
        time.sleep(2)
        # writer_1.add_scalar('test1',i+2,i)
    
    for i in range(10):
        logger.add_scalar('test1','macha1',i*2,i)
        print('adding 2 ',i)
        time.sleep(2)

if __name__ == '__main__':
    test()