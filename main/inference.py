def generate(model,inputs):
    model.eval()
    return model(*inputs)
