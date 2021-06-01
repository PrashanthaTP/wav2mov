@ECHO OFF
TITLE wav2mov
echo Running preprocessing 

set DEVICE="cpu"
set VERSION="preprocess_500_a23456"
set LOG="y"
python main/main.py --preprocess=y  --device=%DEVICE% --version=%VERSION% --log=%LOG% 