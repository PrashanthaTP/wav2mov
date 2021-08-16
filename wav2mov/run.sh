#!/bin/bash
# run main/main.py
EPOCHS=300
NUM_VIDEOS=120
VERSION="v16_sync_expert"
TRAIN_SYNC_EXPERT='n'
COMMENT='GPU | 225 to 300| no noise encoder | interleaved sync training | only l1 at the beginning'
# COMMENT='GPU | 375 to 450| seq upper half and sync lower half | sync bce loss'
DEVICE="cuda"
IS_TRAIN="y"
LOG='y'
MODEL_PATH="/content/drive/MyDrive/Colab Notebooks/wav2mov-dev_phase_8/wav2mov/runs/${VERSION}/Run_11_7_2021__17_44"
#echo 'options chosen are '"$EPOCHS"' is_training = '"$IS_TRAIN"
# python main/main.py --train=$IS_TRAIN -e=$EPOCHS -v=$NUM_VIDEOS -m="$COMMENT" --device=$DEVICE --version=$VERSION --log=$LOG --train_sync_expert=${TRAIN_SYNC_EXPERT}
python main/main.py --train=$IS_TRAIN -e=$EPOCHS -v=$NUM_VIDEOS -m="$COMMENT" --device=$DEVICE --version=$VERSION --log=$LOG --model_path="$MODEL_PATH" --train_sync_expert=${TRAIN_SYNC_EXPERT}
