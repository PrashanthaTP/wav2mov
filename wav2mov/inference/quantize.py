import os
import torch
import torch.quantization
from torch import nn
# here is our floating point instance
from inference.model_utils import get_model
# import the modules used here in this recipe


def print_size_of_model(m, label=""):
    torch.save(m.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

DIR = os.path.dirname(os.path.abspath(__file__))
model = get_model()
target_path =r'models/checkpoints/gen_quantized.pt' 
# this is the call that does the work
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Conv1d,nn.Conv2d,nn.ConvTranspose2d, nn.Linear,nn.GRU}, dtype=torch.qint8
)

# compare the sizes
# f=print_size_of_model(float_lstm,"fp32")
f=print_size_of_model(model,"fp32")
q=print_size_of_model(model_quantized,"int8")
print("{0:.2f} times smaller".format(f/q))
