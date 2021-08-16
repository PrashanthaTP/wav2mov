#!/bin/bash
# run main/main.py
EPOCHS=450
NUM_VIDEOS=120
VERSION="sync_expert"
TRAIN_SYNC_EXPERT='y'
COMMENT='GPU |350 to 450| train only sync disc to make it a expert'
# COMMENT='GPU | 375 to 450| seq upper half and sync lower half | sync bce loss'
DEVICE="cuda"
IS_TRAIN="y"
LOG='y'
# MODEL_PATH="/content/drive/MyDrive/Colab Notebooks/wav2mov-dev_phase_8/wav2mov/runs/v16_sync_expert/Run_10_7_2021__23_46"
# MODEL_PATH="/content/drive/MyDrive/Colab Notebooks/wav2mov-dev_phase_8/wav2mov/runs/${VERSION}/Run_10_7_2021__14_19"
MODEL_PATH="/content/drive/MyDrive/Colab Notebooks/wav2mov-dev_phase_8/wav2mov/runs/${VERSION}/Run_11_7_2021__11_24"
#echo 'options chosen are '"$EPOCHS"' is_training = '"$IS_TRAIN"
# python main/main.py --train=$IS_TRAIN -e=$EPOCHS -v=$NUM_VIDEOS -m="$COMMENT" --device=$DEVICE --version=$VERSION --log=$LOG --train_sync_expert=${TRAIN_SYNC_EXPERT}
python main/main.py --train=$IS_TRAIN -e=$EPOCHS -v=$NUM_VIDEOS -m="$COMMENT" --device=$DEVICE --version=$VERSION --log=$LOG --model_path="$MODEL_PATH" --train_sync_expert=${TRAIN_SYNC_EXPERT}
