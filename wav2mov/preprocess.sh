echo "Preprocess"
DEVICE='cpu'
VERSION='preprocess_500_a23456'
LOG='y'
GRID_DATASET_DIR='/content/drive/MyDrive/Colab Notebooks/projects/wav2mov/datasets/grid_a5_500_a10to14_raw'
python main/main.py --preprocess=y -grid="${GRID_DATASET_DIR}" --device=${DEVICE} --version=${VERSION} --log=${LOG} 
