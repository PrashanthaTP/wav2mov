import os
import random
from tqdm import tqdm

FILENAME = 'filenames.txt'
FILENAME_TRAIN = 'filenames_train.txt'
FILENAME_TEST = 'filenames_test.txt'
DATASET = 'grid_dataset_256_256'
FROM_DIR = r''
TO_DIR = r''
if len(FROM_DIR)==0:
    FROM_DIR = os.path.dirname(os.path.abspath(__file__))
    FROM_DIR = os.path.join(FROM_DIR,DATASET)
    TO_DIR = FROM_DIR

def get_folders_list():
    folders = [folder for _,folder,_ in os.walk(FROM_DIR)][0]
    folders_strs = []
    for folder in tqdm(folders):
        if os.path.isdir(os.path.join(FROM_DIR,folder)):
            folders_strs.append(folder+'\n')
    return folders_strs
    
def write_to_text_file(folders_list):
    with open(os.path.join(TO_DIR,FILENAME),'w') as file:
        file.writelines(folders_list)
    print(f'list written to {os.path.join(TO_DIR,FILENAME)}')


def write_to_file(file_path,content_list):
    with open(file_path,'w') as file:
        file.writelines(content_list)
        
def create_train_test_filelist(folders_list):
    total = len(folders_list)
    random.shuffle(folders_list)
    n_train = int(0.9*total)
    n_test = total-n_train
    write_to_file(os.path.join(TO_DIR,FILENAME_TRAIN),folders_list[:n_train])
    write_to_file(os.path.join(TO_DIR,FILENAME_TEST),folders_list[n_train:])

def main():
    folders = get_folders_list()
    # write_to_text_file(folders)
    create_train_test_filelist(folders)
    
if __name__ == '__main__':
    main()