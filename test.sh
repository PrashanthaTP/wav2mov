#!/bin/bash

echo $1
is_test='y'
model_path='/content/drive/MyDrive/Colab Notebooks/projects/wav2mov_engine/wav2mov/runs/v10/'$1'/gen_'$1'.pt'
log='n'
device='cpu'
version='v10'

python main/main.py --device=${device} --test=${is_test} --version=${version} --model_path="${model_path}" --log=${log} -v=14