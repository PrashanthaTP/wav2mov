@ECHO OFF
TITLE wav2mov
echo Running training script

set EPOCHS=1
set NUM_VIDEOS=14
set VERSION="v9"
set "COMMENT='GPU | scaled id ,gen_id losses by FRACTION | prelearning till 15 epoch | 10 l1_l and 1 id_l'"
set DEVICE="cuda"
set IS_TRAIN="y"
set LOG="n"
set "MODEL_PATH='/content/drive/MyDrive/Colab Notebooks/projects/wav2mov_engine/wav2mov/runs/v9/Run_6_5_2021__17_28'"
@REM echo 'options chosen are '"$EPOCHS"' is_training = '"$IS_TRAIN"
@REM #python main/main.py --train=%IS_TRAIN% -e=%EPOCHS% -v=%NUM_VIDEOS% -m="%COMMENT%" --device=%DEVICE% --version=%VERSION% --log=%LOG% --model_path=%MODEL_PATH%
python main/main.py --train=%IS_TRAIN% -e=%EPOCHS% -v=%NUM_VIDEOS% -m="%COMMENT%" --device=%DEVICE% --version=%VERSION% --log=%LOG%
