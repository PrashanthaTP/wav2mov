import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description='Wav2Mov | End to End Speech to facial animation model')
        
        self.parser.add_argument('--log',default='y',
                            choices=['y','n','yes','no'],
                            type=str,help='whether to initialize logger')
        
        self.parser.add_argument('--train',default='n',
                            choices=['y','n','yes','no'],
                            type=str,help='whether to train the model')
        
        self.parser.add_argument('--test',default='n',
                            choices=['y','n','yes','no'],
                            type=str,help='run test script')
        
        self.parser.add_argument('--preprocess',default='n',
                            choices=['y','n','yes','no'],
                            type=str,help='run preprocess script')
        
        self.parser.add_argument('--device',default='cuda',
                            choices=['cpu','cuda'],
                            type=str,help='device where processing should be done')
        
        self.parser.add_argument('--model_path','-path',
                            type=str,help='generator checkpoint fullpath')
        
        self.parser.add_argument('--num_videos','-v',
                            type=int,help='num of videos on which the model should be trained')
        
        
        self.parser.add_argument('--msg','-m',type=str,help='any message about current run')
        
    def parse(self):
        return self.parser.parse_args()
    