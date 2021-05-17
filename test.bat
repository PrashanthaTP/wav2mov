@ECHO OFF
TITLE Wav2Mov

set is_test=y
set "model_path=E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\wav2mov\wav2mov\runs\v6\%1\gen_%1.pt"
set log=n
set device=cpu
set version=v9
echo %model_path%
python main/main.py --device=%device% --test=%is_test% --version=%version% --model_path=%model_path% --log=%log% -v=14