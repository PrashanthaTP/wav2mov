#!/bin/bash

VERSION="v15_sync_expert"
echo $1
is_test='y'
model_path="/content/drive/MyDrive/Colab Notebooks/wav2mov-dev_phase_6/wav2mov/runs/${VERSION}/${1}/gen_${1}.pt"
log='n'
device='cpu'

python main/main.py --device=${device} --test=${is_test} --version=${VERSION} --model_path="${model_path}" --log=${log} -v=14
