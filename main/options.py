import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              description='Wav2Mov | End to End Speech to facial animation model')
        
        self.parser.add_argument('--log',
                                 default='y',
                                choices=['y','n','yes','no'],
                                type=str,
                                help='whether to initialize logger')
            
        self.parser.add_argument('--train',
                                 default='n',
                                choices=['y','n','yes','no'],
                                type=str,
                                help='whether to train the model')
        
        self.parser.add_argument('--device','-d',
                                default='cpu',
                                choices=['cpu','cuda'],
                                type=str,
                                help='device on which model operations are done',
                                required=True)
        
        self.parser.add_argument('--num_epochs','-e',
                                type=int,
                                help='device on which model operations are done',
                         )
        
        self.parser.add_argument('--test',
                                 default='n',
                                choices=['y','n','yes','no'],
                                type=str,
                                help='run test script')
                
        self.parser.add_argument('--preprocess',
                                 default='n',
                                choices=['y','n','yes','no'],
                                type=str,help='run preprocess script')
        
        # self.parser.add_argument('--device',
        #                          default='cuda',
        #                         choices=['cpu','cuda'],
        #                         type=str,help='device where processing should be done')
        
        self.parser.add_argument('--model_path','-path',
                                type=str,
                                help='generator checkpoint fullpath')
        
        self.parser.add_argument('--num_videos','-v',
                                    type=int,
                                    help='num of videos on which the model should be trained')
        
        
        self.parser.add_argument('--msg','-m',
                                 type=str,
                                 help='any message about current run')
        
    def parse(self):
        return self.parser.parse_args()
    
def set_device(options, params):
    device = options.device
    params.set('device', device)
    params.set('gen', {**params['gen'], 'device': device})
    
def set_epochs(options,params):
    num_epochs = options.num_epochs
    params.set('num_epochs',num_epochs)
    
def set_options(options, params):
    set_device(options,params)
    set_epochs(options,params)
