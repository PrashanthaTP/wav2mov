import torch 
import os 
def save_checkpoint(model,loss_val,location,save_params=True):
    statedict = {
        'model_state_dict': model.state_dict(),
        'min_val_loss' : loss_val,
        'model_arch': str(model)
        }
    os.makedirs(os.path.dirname(location),exist_ok=True)
    torch.save(statedict,location)
    
    if save_params:
        hparams_dir = os.path.dirname(location)
        hparams_filename = 'hparams_' + os.path.basename(location).split('.')[0] + '.json'
        model.hparams.to_json(os.path.join(hparams_dir,hparams_filename))
    # with open(os.path.join(hparams_dir,hparams_filename),'a+') as file:
    #     json.dump(model.hparams.asdict(),)
        
    
    

def load_checkpoint(location):
    return torch.load(location)

def save_fully_trained_model(model,loss_val,location,save_params=True):
    basename,file_extension = os.path.basename(location).split('.')
    new_name = basename +'_fully_trained.' + file_extension
    new_location = os.path.join(os.path.dirname(location),new_name)
    
    save_checkpoint(model,loss_val,new_location,save_params=save_params)
    