@ECHO OFF
TITLE wav2mov
echo Running preprocessing 

set DEVICE="cpu"
set VERSION="preprocess_500_a23456"
set LOG="y"
set GRID_DATASET_DIR="/content/drive/MyDrive/Colab Notebooks/projects/wav2mov/datasets/grid_a5_500_a10to14_raw"
echo %GRID_DATASET_DIR%
@REM python main/main.py --preprocess=y -grid=%GRID_DATASET_DIR% --device=%DEVICE% --version=%VERSION% --log=%LOG% 