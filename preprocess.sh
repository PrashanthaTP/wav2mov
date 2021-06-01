echo "Preprocess"
DEVICE='cpu'
VERSION='preprocess_500_a23456'
LOG='y'
python main/main.py --preprocess=y  --device=$DEVICE --version=$VERSION --log=$LOG 