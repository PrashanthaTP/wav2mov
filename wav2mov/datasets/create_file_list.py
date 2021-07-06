import os
import random
from tqdm import tqdm

FILENAME = 'filenames.txt'
FILENAME_TRAIN = 'filenames_train.txt'
FILENAME_TEST = 'filenames_test.txt'
DATASET = 'grid_dataset_a5_500_a10to14'
FROM_DIR = ''
TO_DIR = ''

VIDEOS_PER_ACTOR = 30

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
      
def get_actors(folders_list):
  actors = set()
  for folder in folders_list:
    actors.add(folder.split('_')[0])
  return sorted(actors)

def get_train_test_list(folders_list):
    actors = get_actors(folders_list)
    train_actors = set(actors[:-1])
    train_dict = {}
    test_list = []
    for folder in folders_list:
      actor = folder.split('_')[0]
      if actor in train_actors:
        if len(train_dict.get(actor,[]))<VIDEOS_PER_ACTOR:
          train_dict.setdefault(actor,[]).append(folder)
      else:
          test_list.append(folder)
    
    train_list = []
    for val in train_dict.values():
      train_list.extend(val)
    return train_list,test_list

def create_train_test_filelist(folders_list):
    train_list,test_list =get_train_test_list(folders_list)
    random.shuffle(train_list)
    write_to_file(os.path.join(TO_DIR,FILENAME_TRAIN),train_list)
    write_to_file(os.path.join(TO_DIR,FILENAME_TEST),test_list)

def main():
    folders = get_folders_list()
    create_train_test_filelist(folders)
    
if __name__ == '__main__':
    main()
