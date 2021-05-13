import torch
def get_input_of_shape(shape):
    return torch.randn(shape)


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            out = func(*args, **kwargs)
            return out
    return wrapper
