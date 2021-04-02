def check_batchsize(hparams):
    if hparams['batch_size']%hparams['mini_batch_size']:
        raise ValueError(f'Batch size must be evenly divisible by mini_batch_size\n'
                         f'Currently batch_size : {hparams["batch_size"]} mini_batch_size :{hparams["mini_batch_size"]}')